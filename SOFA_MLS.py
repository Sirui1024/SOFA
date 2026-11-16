"""MLS-only SOFA: learned candidate selection and local search, no DEA."""
from functools import partial
from sofa_ablation import run_ablation
sofa_parallel = partial(run_ablation, mode='mls')
