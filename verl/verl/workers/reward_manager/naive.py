# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import concurrent.futures
import threading
import os

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.print_lock = threading.Lock()  # Lock for thread-safe printing

    def _process_single_item(self, i, data_item, already_print_data_sources):
        """Process a single data item and return the results."""
        prompt_ids = data_item.batch["prompts"]

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

        data_source = data_item.non_tensor_batch[self.reward_fn_key]

        extra_info = data_item.non_tensor_batch.get("extra_info", None)

        score = self.compute_score(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        reward_extra_info_item = {}
        if isinstance(score, dict):
            reward = score["score"]
            # Store the information including original reward
            reward_extra_info_item = score
        else:
            reward = score

        # Handle printing with lock to avoid interleaved output
        should_print = False
        with self.print_lock:
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                should_print = True
        
        if should_print:
            with self.print_lock:
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        return {
            "i": i,
            "reward": reward,
            "valid_response_length": valid_response_length,
            "reward_extra_info_item": reward_extra_info_item
        }

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Use ThreadPoolExecutor to parallelize processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all tasks and store futures with their indices
            future_to_index = {
                executor.submit(self._process_single_item, i, data[i], already_print_data_sources): i
                for i in range(len(data))
            }
            
            # Process results as they complete, maintaining original order
            results = [None] * len(data)
            for future in concurrent.futures.as_completed(future_to_index):
                result = future.result()
                results[result["i"]] = result
            
            # Apply results in the correct order
            for result in results:
                i = result["i"]
                reward = result["reward"]
                valid_response_length = result["valid_response_length"]
                reward_extra_info_item = result["reward_extra_info_item"]
                
                # Store reward in tensor
                reward_tensor[i, valid_response_length - 1] = reward
                
                # Store extra info if available
                for key, value in reward_extra_info_item.items():
                    reward_extra_info[key].append(value)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
