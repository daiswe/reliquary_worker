"""Protocol-critical constants for the rollout worker.

Pruned subset of reliquary.constants — only values needed for generation,
local reward, and GRAIL commit construction.
"""

import os as _os

GRAIL_PROOF_VERSION = "v7"

PRIME_Q = 2_147_483_647
CHALLENGE_K = 32
RNG_LABEL = {"sketch": b"sketch", "open": b"open", "sat": b"sat"}
LAYER_INDEX = -1

PROOF_BATCH_SIZE = 16
PROOF_TOPK = 16
PROOF_NUM_BUCKETS = 8
PROOF_COEFF_RANGE = 127
PROOF_SKETCH_TOLERANCE_BASE = 5000
PROOF_SKETCH_TOLERANCE_GROWTH = 5.0

ATTN_IMPLEMENTATION = _os.environ.get("GRAIL_ATTN_IMPL", "flash_attention_2")

WINDOW_LENGTH = 5

MAX_NEW_TOKENS_PROTOCOL_CAP = 32768

BFT_ENABLED = True
BFT_THINKING_BUDGET = 2048
BFT_ANSWER_BUDGET = 512
BFT_FORCE_TEMPLATE = "</think>\n\nFinal Answer: \\boxed{"

SHAPE_PENALTY = 0.5
SHAPE_LEN_FRAC = 0.5
MAX_TRUNCATED_PER_SUBMISSION = 1

M_ROLLOUTS = 8

T_PROTO = 0.6
TOP_P_PROTO = 0.95
TOP_K_PROTO = 20

FORCED_SEED_DOMAIN = "reliquary-forced-seed-v1"
FORCED_SEED_PROTOCOL_VERSION = 1
