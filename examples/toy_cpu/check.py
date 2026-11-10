"""The toy project's own test: the kernel cost stays positive and finite."""
import math
import sys

import kernels

ok = all(math.isfinite(kernels.attention_cost(s)) and kernels.attention_cost(s) > 0 for s in (128, 512, 2048))
sys.exit(0 if ok else 1)
