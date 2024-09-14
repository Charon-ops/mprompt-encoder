conda activate mprompt
export OLLAMA_HOST=127.0.0.1:11434 # port 11434,11435
export CUDA_VISIBLE_DEVICES=0 # 0, 1, 2, 3
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=4
ollama serve