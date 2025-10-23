# SuperCoder: Assembly Program Superoptimization with Large Language Models

![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2502.12466-b31b1b.svg)](https://arxiv.org/abs/2505.11480)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Superoptimizer_Qwen7B-orange.svg)](https://huggingface.co/LLM4Code/Superoptimizer_Qwen7B)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-llm_superoptimizer_ds-orange.svg)](https://huggingface.co/datasets/LLM4Code/llm_superoptimizer_ds)





## Setup
1. **Create and activate a Conda environment**:

   ```bash
   conda create -y -n LLMSuperopt python=3.10.12
   conda activate LLMSuperopt
   ```

2. **Install dependencies**:

   ```bash
   bash scripts/install_dependencies.sh
   ```
3. **Install hyperfine**:

   Refer to the [hyperfine](https://github.com/sharkdp/hyperfine) repository for installation instructions.
   
4. **Set API keys**:
   Ensure you have valid API keys for the required services:

   ```bash
   export OPENAI_API_KEY=<your_openai_api_key>
   export ANTHROPIC_API_KEY=<your_anthropic_api_key>
   export TOGETHER_API_KEY=<your_together_api_key>
   export GEMINI_API_KEY=<your_gemini_api_key>
   ```
### Setting up HuggingFace Account

1. **Obtain an access token**:
   - Go to [HuggingFace Tokens](https://huggingface.co/settings/tokens) and generate a token with `read` or `write` permissions.

2. **Login using the token**:

   **Option A**: Use the command line:

   ```bash
   huggingface-cli login
   huggingface-cli whoami
   ```

   **Option B**: Add the token to the environment variable:

   ```plaintext
   export HF_TOKEN=<your_huggingface_token>
   ```
3. **Setting HF_CACHE**:

   ```bash
   export HF_CACHE=~/.cache/huggingface/
   ```
   Please customize your HF_CACHE path.

## RL Training 
We leverage the [verl](https://github.com/volcengine/verl) library for RL training and implement custom reward functions for our superoptimization task.

### Downloading the parquet files 
Download the parquet files locally by running 
```bash
python download_parquet.py --ds_path LLM4Code/llm_superoptimizer_ds
```

### Run the training script

Refer to `verl/examples/ppo_trainer/run_qwen2.57_superopt.sh` for an example training script. The script trains Qwen2.5-Coder-7B-Instruct on the train split of the `llm_superoptimizer_ds` dataset for 1 epoch. Run the script with 
```bash
bash verl/examples/ppo_trainer/run_qwen2.57_superopt.sh
```


## Evaluation 

### Running Main Evaluation
```python
python src/evaluate.py --model_name LLM4Code/Superoptimizer_Qwen7B
```

We support OpenAI models, Anthropic models, Google models, models served by Together.ai, and models hosted on HuggingFace. 

Furthermore, we provide a script that evaluates the performance of different models on the validation split of the `llm_superoptimizer_ds` dataset. Run the script with 
```bash
bash scripts/eval.sh
```
Refer to `src/evaluate.py` for the various arguments used in the evaluation script, for example, in order to evaluate different models.

### Analyzing the results 
Results are saved in the `results` directory. Analyze the results by running 
```bash
python print_results.py results
```


## Custom Training Data Creation 

### Downloading CodeNet and TestCases

Download and unzip the full CodeNet dataset from [here](https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz) and the metadata from [here](https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet_metadata.tar.gz). The unzipped folders should be in the `LLMCompiler/Project_CodeNet` directory. 

Then, download and unzip the test cases from [here](https://drive.google.com/file/d/1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr/view). The unzipped folder should be in the `LLMCompiler/` directory and be named `test_cases`.


### Creating the training data 
Refer to the following script for an example of how to create the data with training and validation splits.
```bash
bash scripts/create_ds.sh
```
Arguments such as the number of problems to consider can be changed in the script. Refer to `src/create_data` for more details.


## Citation

If you use this repository in your research, please cite the corresponding paper:

```bibtex
@misc{wei2025supercoderassemblyprogramsuperoptimization,
      title={SuperCoder: Assembly Program Superoptimization with Large Language Models}, 
      author={Anjiang Wei and Tarun Suresh and Huanmi Tan and Yinglun Xu and Gagandeep Singh and Ke Wang and Alex Aiken},
      year={2025},
      eprint={2505.11480},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.11480}, 
}
```

---

## License

This project is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).




