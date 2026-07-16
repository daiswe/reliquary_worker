"""Rollout generation engine — model load, forced-seed sampling, GRAIL commits."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import TYPE_CHECKING

from worker.constants import (
    FORCED_SEED_PROTOCOL_VERSION,
    LAYER_INDEX,
    MAX_NEW_TOKENS_PROTOCOL_CAP,
    M_ROLLOUTS,
)
from worker.protocol.submission import (
    UnsignedRolloutSubmission,
    WorkerNextResponse,
    WorkerSubmitRequest,
)

if TYPE_CHECKING:
    from worker.environment.base import Environment

logger = logging.getLogger(__name__)


async def maybe_pull_checkpoint(
    assignment: WorkerNextResponse,
    local_n: int,
    local_hash: str,
    local_model,
    *,
    download_fn,
    load_fn,
):
    """If remote checkpoint_n > local, download via HF and load."""
    if assignment.checkpoint_n <= local_n:
        return local_n, local_hash, local_model
    if assignment.checkpoint_repo_id is None or assignment.checkpoint_revision is None:
        return local_n, local_hash, local_model
    local_path = await download_fn(
        assignment.checkpoint_repo_id, assignment.checkpoint_revision,
    )
    new_model = load_fn(local_path)
    return assignment.checkpoint_n, assignment.checkpoint_revision, new_model


async def _hf_download(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download
    from worker.shared.modeling import MODEL_SNAPSHOT_ALLOW_PATTERNS

    return await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
    )


def _compute_merkle_root(rollouts) -> str:
    leaves = []
    for i, r in enumerate(rollouts):
        h = hashlib.sha256()
        h.update(i.to_bytes(8, "big"))
        h.update(json.dumps(r.tokens, separators=(",", ":")).encode())
        h.update(json.dumps(r.reward).encode())
        h.update(json.dumps(r.commit, sort_keys=True, separators=(",", ":")).encode())
        leaves.append(h.digest())

    while len(leaves) > 1:
        new = []
        for i in range(0, len(leaves), 2):
            left = leaves[i]
            right = leaves[i + 1] if i + 1 < len(leaves) else left
            new.append(hashlib.sha256(left + right).digest())
        leaves = new
    return leaves[0].hex()


def _bft_assemble_rollouts(
    *, model, phase1_tensor, prompt_tokens, think_close_ids, force_ids,
    eos_ids, answer_budget, randomness, hotkey, prompt_idx, checkpoint_hash,
    gen_kwargs=None,
):
    import torch

    from worker.forced_seed_sampler import (
        ForcedSeedLogitsProcessor, forced_seed_generate_kwargs, phase2_base_offsets,
    )
    from worker.shared.modeling import first_eos_index, has_think_close

    plen = len(prompt_tokens)
    n = int(phase1_tensor.shape[0])
    close_set = {int(t) for t in think_close_ids}
    force_ids = [int(t) for t in force_ids]

    out: list = [None] * n
    unfinished_idx: list[int] = []
    unfinished_primed: list[list[int]] = []
    unfinished_force_spans: list[tuple[int, int] | None] = []
    for i in range(n):
        seq = phase1_tensor[i].tolist()
        gen = seq[plen:]
        fe = first_eos_index(gen, eos_ids)
        if fe is not None:
            gen = gen[: fe + 1]
            out[i] = {"tokens": prompt_tokens + gen,
                      "prompt_length": plen, "forced": False}
        elif has_think_close(gen, close_set):
            unfinished_idx.append(i)
            unfinished_primed.append(seq)
            unfinished_force_spans.append(None)
        else:
            force_start = len(seq)
            primed = seq + force_ids
            unfinished_idx.append(i)
            unfinished_primed.append(primed)
            unfinished_force_spans.append((force_start, force_start + len(force_ids)))

    if unfinished_primed:
        width = max(len(p) for p in unfinished_primed)
        pad = min(eos_ids) if eos_ids else 0
        rows = [[pad] * (width - len(p)) + p for p in unfinished_primed]
        mask = [[0] * (width - len(p)) + [1] * len(p) for p in unfinished_primed]
        device = getattr(model, "device", "cpu")
        proc = ForcedSeedLogitsProcessor(
            randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
            checkpoint_hash=checkpoint_hash,
            rollout_indices=list(unfinished_idx),
            base_offsets=phase2_base_offsets(
                [len(p) for p in unfinished_primed], plen,
            ),
            start_len=width,
        )
        ans = model.generate(
            torch.tensor(rows, device=device),
            attention_mask=torch.tensor(mask, device=device),
            max_new_tokens=answer_budget,
            **forced_seed_generate_kwargs(gen_kwargs or {}, proc),
        )
        for k, i in enumerate(unfinished_idx):
            primed = unfinished_primed[k]
            tail = ans[k].tolist()[width:]
            fe = first_eos_index(tail, eos_ids)
            tail = tail[: fe + 1] if fe is not None else tail
            forced_span = unfinished_force_spans[k]
            rollout = {"tokens": primed + tail, "prompt_length": plen,
                       "forced": forced_span is not None}
            if forced_span is not None:
                rollout["force_span"] = forced_span
            out[i] = rollout
    return out


def _rollout_metadata(generation: dict, token_logprobs: list) -> dict:
    prompt_length = int(generation["prompt_length"])
    all_tokens = generation["tokens"]
    force_span = generation.get("force_span")
    return {
        "prompt_length": prompt_length,
        "completion_length": len(all_tokens) - prompt_length,
        "success": True,
        "total_reward": 0.0,
        "advantage": 0.0,
        "token_logprobs": token_logprobs,
        "forced": bool(generation.get("forced", False)),
        "force_span": list(force_span) if force_span else None,
    }


class RolloutEngine:
    """GPU rollout generation: vLLM (GPU 0) for generation, HF (GPU 1) for proofs."""

    def __init__(
        self,
        vllm_model,
        hf_model,
        tokenizer,
        envs: dict[str, "Environment"],
        *,
        vllm_gpu: int = 0,
        proof_gpu: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS_PROTOCOL_CAP,
    ) -> None:
        self.vllm_model = vllm_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.envs = envs
        self.vllm_gpu = vllm_gpu
        self.proof_gpu = proof_gpu
        self.max_new_tokens = max_new_tokens
        self._loaded_checkpoint_path: str | None = None

        from worker.shared.hf_compat import resolve_hidden_size
        from worker.protocol.grail_verifier import GRAILVerifier

        self._hidden_dim = resolve_hidden_size(hf_model)
        self._verifier = GRAILVerifier(hidden_dim=self._hidden_dim)

    async def run_prompt(self, assignment: WorkerNextResponse) -> WorkerSubmitRequest:
        """Generate M rollouts for one orchestrator assignment."""
        env = self.envs[assignment.env_name]
        problem = env.get_problem(assignment.prompt_idx)
        checkpoint_hash = assignment.checkpoint_hash or assignment.checkpoint_revision or ""

        generations = self._generate_m_rollouts(
            problem,
            assignment.randomness,
            env_name=assignment.env_name,
            prompt_idx=assignment.prompt_idx,
            checkpoint_hash=checkpoint_hash,
            miner_hotkey=assignment.miner_hotkey,
        )
        if len(generations) < M_ROLLOUTS:
            raise RuntimeError(
                f"generated {len(generations)}/{M_ROLLOUTS} rollouts "
                f"for prompt {assignment.prompt_idx}"
            )

        rollout_submissions = [
            self._build_rollout_submission(
                gen, problem, assignment.randomness, env=env,
            )
            for gen in generations
        ]
        merkle_root = _compute_merkle_root(rollout_submissions)

        return WorkerSubmitRequest(
            prompt_idx=assignment.prompt_idx,
            window_n=assignment.window_n,
            env_name=assignment.env_name,
            merkle_root=merkle_root,
            rollouts=rollout_submissions,
        )

    def _load_checkpoint(self, local_path: str):
        import torch

        from worker.constants import ATTN_IMPLEMENTATION
        from worker.shared.modeling import load_text_generation_model

        if self._loaded_checkpoint_path == local_path:
            return self.hf_model

        logger.info("Loading checkpoint from %s", local_path)

        try:
            new_hf = load_text_generation_model(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.proof_gpu}").eval()
        except Exception:
            logger.exception("Failed to reload hf_model from %s", local_path)
            return self.hf_model

        old_hf = self.hf_model
        self.hf_model = new_hf
        del old_hf
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            new_gen = load_text_generation_model(
                local_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=ATTN_IMPLEMENTATION,
            ).to(f"cuda:{self.vllm_gpu}").eval()
        except Exception:
            logger.exception("Failed to reload vllm_model from %s", local_path)
            self.vllm_model = None
            self._loaded_checkpoint_path = None
            return self.hf_model

        old_gen = self.vllm_model
        self.vllm_model = new_gen
        del old_gen
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        self._loaded_checkpoint_path = local_path
        return self.hf_model

    def _generate_m_rollouts(
        self, problem, randomness, *, env_name: str | None = None,
        prompt_idx: int, checkpoint_hash: str, miner_hotkey: str,
    ) -> list[dict]:
        import torch

        from worker.constants import (
            BFT_ANSWER_BUDGET,
            BFT_ENABLED,
            BFT_THINKING_BUDGET,
        )
        from worker.forced_seed_sampler import (
            ForcedSeedLogitsProcessor, forced_seed_generate_kwargs,
        )
        from worker.protocol.tokens import encode_prompt
        from worker.shared.modeling import (
            first_eos_index,
            force_close_token_ids,
            resolve_eos_token_ids,
            think_close_token_ids,
        )

        hotkey = miner_hotkey
        prompt_tokens = encode_prompt(self.tokenizer, problem["prompt"])
        prompt_length = len(prompt_tokens)
        eos_ids = resolve_eos_token_ids(self.vllm_model, self.tokenizer)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and eos_ids:
            pad_token_id = min(eos_ids)
        bft_applicable = BFT_ENABLED and (
            env_name is None or env_name == "openmathinstruct"
        )

        with torch.no_grad():
            input_tensor = torch.tensor(
                [prompt_tokens] * M_ROLLOUTS,
                device=getattr(self.vllm_model, "device", "cpu"),
            )
            attention_mask = torch.ones_like(input_tensor)
            base_kwargs = {
                "max_new_tokens": (
                    min(self.max_new_tokens, BFT_THINKING_BUDGET)
                    if bft_applicable else self.max_new_tokens
                ),
                "pad_token_id": pad_token_id,
                "attention_mask": attention_mask,
            }
            if eos_ids:
                base_kwargs["eos_token_id"] = sorted(eos_ids)
            phase1_proc = ForcedSeedLogitsProcessor(
                randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                checkpoint_hash=checkpoint_hash,
                rollout_indices=list(range(M_ROLLOUTS)),
                base_offsets=[0] * M_ROLLOUTS, start_len=prompt_length,
            )
            outputs = self.vllm_model.generate(
                input_tensor,
                **forced_seed_generate_kwargs(base_kwargs, phase1_proc),
            )

            if bft_applicable:
                phase2_kwargs = {"pad_token_id": pad_token_id}
                if eos_ids:
                    phase2_kwargs["eos_token_id"] = sorted(eos_ids)
                return _bft_assemble_rollouts(
                    model=self.vllm_model,
                    phase1_tensor=outputs,
                    prompt_tokens=prompt_tokens,
                    think_close_ids=set(think_close_token_ids(self.tokenizer)),
                    force_ids=force_close_token_ids(self.tokenizer),
                    eos_ids=eos_ids,
                    answer_budget=BFT_ANSWER_BUDGET,
                    randomness=randomness, hotkey=hotkey, prompt_idx=prompt_idx,
                    checkpoint_hash=checkpoint_hash,
                    gen_kwargs=phase2_kwargs,
                )
        rollouts = []
        for i in range(M_ROLLOUTS):
            seq = outputs[i].tolist()
            gen = seq[prompt_length:]
            first_eos = first_eos_index(gen, eos_ids)
            if first_eos is not None:
                gen = gen[: first_eos + 1]
            rollouts.append({
                "tokens": prompt_tokens + gen,
                "prompt_length": prompt_length,
                "forced": False,
            })
        return rollouts

    def _build_rollout_submission(
        self, generation, problem, randomness, *, env=None,
    ) -> UnsignedRolloutSubmission:
        active_env = env if env is not None else next(iter(self.envs.values()))
        all_tokens = generation["tokens"]
        prompt_length = generation["prompt_length"]
        completion_tokens = all_tokens[prompt_length:]
        completion_text = self.tokenizer.decode(completion_tokens)
        if getattr(active_env, "validator_authoritative_reward", False):
            reward = 0.0
        else:
            reward = active_env.compute_reward(problem, completion_text)

        commit = self._build_grail_commit(generation, randomness)
        return UnsignedRolloutSubmission(
            tokens=all_tokens,
            reward=reward,
            commit=commit,
            env_name=active_env.name,
        )

    def _build_grail_commit(self, generation: dict, randomness: str) -> dict:
        """Construct unsigned GRAIL commit — orchestrator signs server-side."""
        import torch

        from worker.constants import GRAIL_PROOF_VERSION
        from worker.shared.forward import forward_single_layer

        all_tokens: list[int] = generation["tokens"]
        prompt_length: int = generation["prompt_length"]

        proof_input = torch.tensor(
            [all_tokens], device=f"cuda:{self.proof_gpu}"
        )
        with torch.no_grad():
            hidden_states, logits = forward_single_layer(
                self.hf_model, proof_input, None, LAYER_INDEX
            )

        hidden_states = hidden_states[0]

        r_vec = self._verifier.generate_r_vec(randomness)
        commitments = self._verifier.create_commitments_batch(hidden_states, r_vec)

        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        token_logprobs: list[float] = []
        for i in range(prompt_length, len(all_tokens)):
            token_logprobs.append(log_probs[i - 1, all_tokens[i]].item())

        model_name: str = getattr(self.hf_model, "name_or_path", "unknown")

        return {
            "tokens": all_tokens,
            "commitments": commitments,
            "proof_version": GRAIL_PROOF_VERSION,
            "model": {"name": model_name, "layer_index": LAYER_INDEX},
            "beacon": {"randomness": randomness},
            "rollout": _rollout_metadata(generation, token_logprobs),
        }
