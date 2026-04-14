docker run -it --gpus all \
-v ~/.cache/huggingface:/root/.cache/huggingface \
-v ~/.cache/vllm:/root/.cache/vllm \
-p 8001:8000 \
--ipc=host \
vllm/vllm-openai \
--model Qwen/Qwen2.5-Coder-0.5B-Instruct