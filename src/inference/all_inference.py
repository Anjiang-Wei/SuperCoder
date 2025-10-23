import requests
from transformers import AutoTokenizer
import os
import ray
from datasets import load_dataset
from sglang.utils import terminate_process, wait_for_server
from sglang.utils import launch_server_cmd
import tempfile
import subprocess
from tqdm import tqdm
import shutil
import psutil
import signal
import concurrent.futures
import json
import numpy as np
import fire
from pathlib import Path
from .inference_clients import get_client
from concurrent.futures import ThreadPoolExecutor, as_completed



class SGLangInference:
    def __init__(self, model_name, temperature, max_new_tokens, sglang_addr, fewshot_questions, fewshot_answers, best_of, eval_type):
        self.model_name = model_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = os.environ['HF_CACHE'], trust_remote_code = True)
        self.sglang_addr = sglang_addr
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens 
        self.fewshot_questions = fewshot_questions
        self.fewshot_answers = fewshot_answers
        self.best_of = best_of
        self.eval_type = eval_type

    def __call__(self, item):
        prompt = item['extra_info'][0]['question']

        if 'instruct' in self.model_name.lower() or 'it' in self.model_name.lower() or 'chat' in self.model_name.lower():
            messages = []
            for question, answer in zip(self.fewshot_questions, self.fewshot_answers):
                messages.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])
            messages.extend([{'role': 'user', 'content': prompt}])
            prompt = self.hf_tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        else:
            prompt = '\n'.join([q + '\n' + a for q, a in zip(self.fewshot_questions, self.fewshot_answers)]) + '\n' + prompt

        sampling_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "n": self.best_of,
        }
        
        json_data = {
        "text": prompt,
        "sampling_params": sampling_params,
        }
        response = requests.post(
                f"{self.sglang_addr}/generate",
                json=json_data,
            )
        
        out = response.json()
        res = []
        if self.best_of > 1:
            for i in range(self.best_of):
                llm_resp = out[i]['text']
                if '```assembly' in llm_resp:
                    llm_resp = llm_resp[llm_resp.rfind('```assembly') + len('```assembly'):]
                if '```' in llm_resp:
                    llm_resp = llm_resp[:llm_resp.rfind('```')]
                res.append({'idx': int(item['idx']), 'sample_id': i, 'output': llm_resp})
        else:
            llm_resp = out['text']
            if '```assembly' in llm_resp:
                llm_resp = llm_resp[llm_resp.rfind('```assembly') + len('```assembly'):]
            if '```' in llm_resp:
                llm_resp = llm_resp[:llm_resp.rfind('```')]
            res.append({'idx': int(item['idx']), 'sample_id': 0, 'output': llm_resp})
            
        return res


def sglang_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, eval_type):

        command_str = f"python -m sglang.launch_server --model-path {model_name} --host 0.0.0.0" 


        server_process, port = launch_server_cmd(command_str)
        wait_for_server(f"http://localhost:{port}")
        sglang_addr = f"http://localhost:{port}"
        
        print(f"Total number of items to process: {len(test_dataset)}")
        
        try:
            ray_dataset = ray.data.from_huggingface(test_dataset)
        
            @ray.remote(num_gpus=0)
            class SGLangInferenceActor:
                def __init__(self):
                    self.inference = SGLangInference(model_name, temperature, max_new_tokens, sglang_addr, fewshot_questions, fewshot_answers, best_of, eval_type)
                
                def process_item(self, item):
                    return self.inference(item)

            if not ray.is_initialized():
                ray.init()

            actors = [SGLangInferenceActor.remote() for _ in range(int(num_workers))]
            print('Number of parallel actors:', len(actors))
            futures = []
            for batch in ray_dataset.iter_batches(batch_size=1):
                actor_id = len(futures) % len(actors)
                futures.append(actors[actor_id].process_item.remote(batch))

            gen_results = []

            with tqdm(total=len(test_dataset), dynamic_ncols=True) as pbar:
                while futures:
                    done, futures = ray.wait(futures, num_returns=1)
                    gen_results.append(ray.get(done[0]))
                    pbar.update(1)

            return gen_results

        except Exception as e:
            raise e
        finally:
            terminate_process(server_process)

def api_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, client_type, eval_type):
    client = get_client(client_type)
    print(f"Total number of items to process: {len(test_dataset)}")

    def build_messages(fewshot_questions, fewshot_answers, prompt):
        messages = []
        for q, a in zip(fewshot_questions, fewshot_answers):
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": prompt})
        return messages

    def process_item(item):
        if eval_type == 'main':
            qkey = 'question'
        elif eval_type == 'c_only':
            qkey = 'c_only_question'
        else:
            raise ValueError(f"Invalid evaluation type: {eval_type}")
        
        prompt = item['extra_info'][qkey]
        messages = build_messages(fewshot_questions, fewshot_answers, prompt)
        # debug
        # print(f"Prompt: {prompt}, Messages: {messages}")
        try:
            completions = client.generate_completion(
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=best_of,
                messages=messages
            )
        except Exception as e:
            print(f"Error in process_item: {e}")
            completions = [""] * best_of
        res = []
        # print(f"completions: {completions}")
        for i, llm_resp in enumerate(completions):
            # Extract assembly code block if present
            if '```assembly' in llm_resp:
                llm_resp = llm_resp[llm_resp.rfind('```assembly') + len('```assembly'):]
            if '```' in llm_resp:
                llm_resp = llm_resp[:llm_resp.rfind('```')]
            # print(f"extracted llm_resp: {llm_resp}")
            res.append({'idx': int(item['idx']), 'sample_id': i, 'output': llm_resp})
        return res

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_item, item) for item in test_dataset]
        with tqdm(total=len(test_dataset), dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                results.append(future.result())
                pbar.update(1)
    return results

def openai_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, eval_type):
    return api_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, client_type="openai", eval_type=eval_type)

def together_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, eval_type):
    return api_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, client_type="together", eval_type=eval_type)

def claude_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, eval_type):
    return api_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, client_type="claude", eval_type=eval_type)

def gemini_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, eval_type):
    return api_generate(test_dataset, model_name, temperature, max_new_tokens, fewshot_questions, fewshot_answers, best_of, num_workers, client_type="gemini", eval_type=eval_type)