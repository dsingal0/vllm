# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class SuffixDecodingProposer:
    """
    Speculative decoding proposer for Suffix Decoding (https://arxiv.org/pdf/2411.04975).
    This class imports and uses the official implementation from Arctic Inference
    (https://github.com/snowflakedb/ArcticInference).
    """

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None, "Speculative config must be set"
        self.num_speculative_tokens = config.num_speculative_tokens
        self.max_tree_depth = config.suffix_decoding_max_tree_depth
        self.max_spec_factor = config.suffix_decoding_max_spec_factor
        self.min_token_prob = config.suffix_decoding_min_token_prob
        self.max_model_len = vllm_config.model_config.max_model_len

        # Lazy import to avoid error when Suffix Decoding is not used.
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        # Initialize and empty cache. This object will take care of caching request
        # outputs, evicting old requests, and manages the per-prompt suffix trees.
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_decoding_max_tree_depth,
            max_cached_requests=config.suffix_decoding_max_cached_requests,
        )

    # --- Scheduler-side helpers for async scheduling ---

    def start_request_scheduler(
        self,
        req_id: str,
        prompt_token_ids: list[int] | torch.Tensor,
    ) -> None:
        """Initialize the suffix cache for a request (scheduler-side)."""
        import numpy as np
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = prompt_token_ids.cpu().numpy()
        if not isinstance(prompt_token_ids, np.ndarray):
            prompt_token_ids = np.array(prompt_token_ids, dtype=np.int32)
        elif prompt_token_ids.dtype != np.int32:
            prompt_token_ids = prompt_token_ids.astype(np.int32)
        self.suffix_cache.start_request(req_id, prompt_token_ids)

    def update_cache_scheduler(
        self,
        req_id: str,
        sampled_token_ids: list[int],
    ) -> None:
        """Update suffix cache with newly sampled tokens (scheduler-side)."""
        self.suffix_cache.add_active_response(req_id, sampled_token_ids)

    def propose_for_request_scheduler(
        self,
        req_id: str,
        all_token_ids: list[int] | np.ndarray | torch.Tensor,
        num_tokens: int,
    ) -> list[int]:
        """Propose draft tokens for a request using current token state.

        Args:
            req_id: The request ID.
            all_token_ids: Full token sequence (prompt + output).
            num_tokens: Total number of tokens in the sequence.
        """
        import numpy as np
        if isinstance(all_token_ids, torch.Tensor):
            all_token_ids = all_token_ids.cpu().numpy()
        if not isinstance(all_token_ids, np.ndarray):
            all_token_ids = np.array(all_token_ids, dtype=np.int32)
        elif all_token_ids.dtype != np.int32:
            all_token_ids = all_token_ids.astype(np.int32)

        start = max(0, num_tokens - self.max_tree_depth)
        pattern = all_token_ids[start:num_tokens]
        draft = self.suffix_cache.speculate(
            req_id,
            pattern,
            max_spec_tokens=min(
                self.num_speculative_tokens,
                self.max_model_len - num_tokens - 1,
            ),
            max_spec_factor=self.max_spec_factor,
            min_token_prob=self.min_token_prob,
        )
        return list(draft.token_ids)

    # --- Original worker-side propose ---

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused
    ) -> list[list[int]]:
        """
        Propose speculative tokens for each request in the input batch. Suffix Decoding
        will speculate a dynamic number of tokens for each request every decoding step,
        so each entry in the returned list may have different lengths.
        """
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                # Skip speculative decoding for partial prefills.
                draft_token_ids.append([])
                continue

            req_id = input_batch.req_ids[i]
            num_tokens = input_batch.num_tokens_no_spec[i]
            if num_tokens >= self.max_model_len:
                # Skip requests that have already reached the max model length.
                draft_token_ids.append([])
                continue

            index = input_batch.req_id_to_index[req_id]
            if req_id not in self.suffix_cache.active_requests:
                if req_id in self.suffix_cache.cached_requests:
                    # Reset the suffix cache for this request.
                    self.suffix_cache.evict_cached_response(req_id)
                num_prompt_tokens = input_batch.num_prompt_tokens[index]
                prompt_token_ids = input_batch.token_ids_cpu[index, :num_prompt_tokens]
                # Start a new request, this will build the suffix tree for that prompt.
                self.suffix_cache.start_request(req_id, prompt_token_ids)

            # Append the newly sampled ids to the suffix cache for this request.
            self.suffix_cache.add_active_response(req_id, sampled_ids)

            # Suffix decoding only uses the most recent tokens up to max_tree_depth, so
            # we extract the pattern from the end of the input.
            start = max(0, num_tokens - self.max_tree_depth)
            pattern = input_batch.token_ids_cpu[i, start:num_tokens]
            draft = self.suffix_cache.speculate(
                req_id,
                pattern,
                max_spec_tokens=min(
                    self.num_speculative_tokens, self.max_model_len - num_tokens - 1
                ),
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            draft_token_ids.append(draft.token_ids)

        # Stop requests that were not seen in the input batch.
        for req_id in (
            self.suffix_cache.active_requests - input_batch.req_id_to_index.keys()
        ):
            self.suffix_cache.stop_request(req_id)

        return draft_token_ids

    def stop_request_scheduler(self, req_id: str) -> None:
        """Stop tracking a request in the suffix cache."""
        self.suffix_cache.stop_request(req_id)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
