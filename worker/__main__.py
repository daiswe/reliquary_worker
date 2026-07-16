"""Rollout worker entrypoint — no CLI flags."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(threadName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def _main() -> None:
    load_dotenv()
    _setup_logging()

    import httpx
    import torch

    from worker.client import fetch_next_assignment, submit_rollouts
    from worker.constants import ATTN_IMPLEMENTATION
    from worker.engine import RolloutEngine, _hf_download, maybe_pull_checkpoint
    from worker.environment import load_environments
    from worker.shared.modeling import (
        MODEL_SNAPSHOT_ALLOW_PATTERNS,
        load_text_generation_model,
        load_tokenizer,
    )

    logger.info(
        "Worker starting (orchestrator=%s:%s)",
        os.environ.get("ORCH_HOST", "127.0.0.1"),
        os.environ.get("ORCH_PORT", "8899"),
    )

    async with httpx.AsyncClient(timeout=60) as client:
        assignment = await fetch_next_assignment(client=client)
        logger.info(
            "First assignment: window=%d prompt=%d env=%s checkpoint_n=%d",
            assignment.window_n, assignment.prompt_idx,
            assignment.env_name, assignment.checkpoint_n,
        )

        if assignment.checkpoint_repo_id and assignment.checkpoint_revision:
            from huggingface_hub import snapshot_download
            initial_path = await asyncio.to_thread(
                snapshot_download,
                repo_id=assignment.checkpoint_repo_id,
                revision=assignment.checkpoint_revision,
                allow_patterns=MODEL_SNAPSHOT_ALLOW_PATTERNS,
            )
        else:
            raise RuntimeError(
                "orchestrator returned no checkpoint — cannot load models"
            )

        logger.info("Loading models from %s", initial_path)
        tokenizer = load_tokenizer(initial_path)
        proof_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"

        vllm_model = load_text_generation_model(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to("cuda:0").eval()

        hf_model = load_text_generation_model(
            initial_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPLEMENTATION,
        ).to(proof_device).eval()

        envs = load_environments([assignment.env_name])
        engine = RolloutEngine(
            vllm_model,
            hf_model,
            tokenizer,
            envs,
            proof_gpu=0 if proof_device == "cuda:0" else 1,
        )
        engine._loaded_checkpoint_path = initial_path

        local_n = assignment.checkpoint_n
        local_hash = assignment.checkpoint_hash or assignment.checkpoint_revision or ""
        loaded_envs: set[str] = set()

        while True:
            if assignment.env_name not in loaded_envs:
                extra = load_environments([assignment.env_name])
                engine.envs.update(extra)
                loaded_envs.add(assignment.env_name)

            if not assignment.randomness:
                logger.debug("waiting for window randomness")
                await asyncio.sleep(0.1)
                assignment = await fetch_next_assignment(client=client)
                continue

            try:
                local_n, local_hash, engine.hf_model = await maybe_pull_checkpoint(
                    assignment,
                    local_n=local_n,
                    local_hash=local_hash,
                    local_model=engine.hf_model,
                    download_fn=_hf_download,
                    load_fn=engine._load_checkpoint,
                )
                assignment.checkpoint_hash = local_hash
            except Exception:
                logger.exception("checkpoint pull failed; keeping local")

            try:
                submit_req = await engine.run_prompt(assignment)
            except Exception:
                logger.exception(
                    "generation failed for prompt %d; fetching next",
                    assignment.prompt_idx,
                )
                assignment = await fetch_next_assignment(client=client)
                continue

            await submit_rollouts(submit_req, client=client)
            logger.info(
                "submitted window=%d prompt=%d merkle=%s",
                submit_req.window_n, submit_req.prompt_idx,
                submit_req.merkle_root[:16],
            )

            assignment = await fetch_next_assignment(client=client)


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
