"""Random-LS SOFA: uniform candidate selection and local search, no MLS/DEA."""
from functools import partial
from sofa_ablation import run_ablation
sofa_parallel = partial(run_ablation, mode='random_ls')
