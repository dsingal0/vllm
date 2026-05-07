python3 /workspace/vllm/measure_startup.py &
export TORCHINDUCTOR_CACHE_DIR=~/.cache/vllm/torchinductor
export VLLM_USE_AOT_COMPILE=0
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_DEEP_GEMM_WARMUP=skip
export TRITON_CACHE_DIR=~/.cache/vllm/triton
vllm serve /workspace/qwen/ \
    --distributed-executor-backend uni \
    --served-model-name Qwen/Qwen3.5-4B \
    --max-num-seqs 16 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gdn-prefill-backend triton \
    --model-impl vllm \
    --load-format runai_streamer \
    --language-model-only
