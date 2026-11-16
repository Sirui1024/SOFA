"""DEA-only SOFA: native evolutionary offspring, no MLS or local search."""
from functools import partial
from sofa_ablation import run_ablation
sofa_parallel = partial(run_ablation, mode='dea')
