#!/bin/bash
# Launch vLLM serving GLM-5.2-FP8blk-bf16attn on 8xB200 with the same
# configuration the customer runs on sglang.
#
# Maps the customer's sglang flags to vLLM equivalents and adds strict
# weight tracking so any missing tensor or FP8 scale raises an error
# instead of being silently dropped.
#
# Requires the parent-module ignored_layers fix in this branch
# (vllm/model_executor/layers/quantization/utils/quant_utils.py) so the
# checkpoint's parent entries (e.g. ``model.layers.0.self_attn``) correctly
# skip all attention submodules and keep them in BF16.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/root/models/baseten_mirendil_glm52_fp8}
PORT=${PORT:-30000}
IMAGE=${IMAGE:-vllm/vllm-openai:latest}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
HF_TOKEN_VALUE=$(cat /root/hf_token 2>/dev/null || echo "${HF_TOKEN:-}")

# Apply the parent-module ignored_layers fix at container launch by
# bind-mounting the patched quant_utils.py over the image's copy (no
# image rebuild needed for quick iteration). When using an image built
# from this branch (via build-vllm-from-source), this mount is a no-op
# because the image already contains the fix.
PATCHED_QUANT_UTILS=/root/dsingal_vllm/vllm/model_executor/layers/quantization/utils/quant_utils.py
CONTAINER_QU=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/quant_utils.py

echo "Launching vLLM $IMAGE on port $PORT with model $MODEL_DIR"

docker run -d --name glm52-vllm \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  -v "$MODEL_DIR:/models/glm52:ro" \
  -v "$PATCHED_QUANT_UTILS:$CONTAINER_QU:ro" \
  -v /root/hf_token:/root/hf_token:ro \
  -e HF_TOKEN="$HF_TOKEN_VALUE" \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  "$IMAGE" \
  --model /models/glm52 \
  --served-model-name glm-5.2-fp8 \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --kv-cache-dtype bfloat16 \
  --moe-backend triton \
  --attention-backend FLASHMLA_SPARSE \
  --hf-overrides '{"head_dtype": "float32"}' \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --max-cudagraph-capture-size 64 \
  --max-model-len "$MAX_MODEL_LEN" \
  --model-loader-extra-config '{"enable_weights_track": true}' \
  --trust-remote-code \
  --host 0.0.0.0 --port "$PORT"

echo "Container started. Logs: docker logs -f glm52-vllm"
echo "Health: curl http://localhost:$PORT/health"
echo "Test:  curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"glm-5.2-fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
