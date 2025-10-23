echo "Starting evaluation of LLM4Code/Superoptimizer_Qwen7B..."
python src/evaluate.py --model_name LLM4Code/Superoptimizer_Qwen7B --num_workers 2

echo "Starting evaluation of claude-3-7-sonnet-latest..."
python src/evaluate.py --inference_engine claude --model_name claude-3-7-sonnet-latest --num_workers 2


echo "Starting evaluation of facebook/llm-compiler-13b..."
python src/evaluate.py --model_name facebook/llm-compiler-13b --num_workers 2



