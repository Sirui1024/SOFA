# SOFA

A research simulator for multi-objective workload routing across four regions.
It evaluates **utility loss, electricity cost, carbon emissions, and water consumption**.

Supported methods: SOFA, NSGA3, SWFAGA, Kimchi, GreenCourier, CASA, fixed-routing
policies, and three SOFA ablations.

## Installation

Use **Python 3.10**. From the repository directory:

```sh
python -m venv .venv
```

Activate the environment:

```sh
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv/Scripts/Activate.ps1
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run a simulation

The synthetic datasets are included and loaded automatically; no extraction is needed.

Start with a lightweight fixed-routing run:

```sh
python 15min_simulator.py -w synthetic -f change -r 6 -e 96 --seed 2 --output-root results
```

Run SOFA with a 900-second search budget:

```sh
python 15min_simulator.py -w synthetic -f SOFA -r 3 -e 96 -t 900 -ff 4 -rs 15 -st 10 -p 30 --seed 2 --output-root results
```

Replace `-f SOFA` with `NSGA3`, `SWFAGA`, `kimchi`, `greenc`, or `CASA`
to run another framework. `-e 96` selects the 96th 15-minute interval, not a
full-day simulation. `-r 6` is a small raw multiplier; `-r 3` selects workload
level 3. Multiprocessing startup can add substantial time, especially on Windows.

Results are saved under `results/<framework>/seed<N>/run_<id>/`.
A successful run includes `run_config.json` and `run_summary.json`.

## Simulator options

These are the defaults in [15min_simulator.py](15min_simulator.py); the examples
above specify their own configurations. Option names and framework names are
case-sensitive. Always specify `-w synthetic` to use the bundled datasets:
the default workload is `compute`, whose files are not included.

| Option | Default | Meaning |
| --- | --- | --- |
| `-h`, `--help` | No | Print the available options and exit. |
| `-w`, `--workload` | `compute` | Workload: `synthetic`, `compute`, `io`, or `c2`. Only `synthetic` has bundled data. `compute` and `io` need external workload pairs; `c2` is accepted but has no workload loader. |
| `-f`, `--framework` | `even` | Routing/search method. See the complete list below. |
| `-o`, `--origin` | `change` | Where requests originate before routing: `even`, `coast`, `centre`, or `change`. This is independent of the routing method. |
| `-e`, `--epoch` | `96` | Select one 15-minute interval, numbered `1` through `96`. Does not run all preceding intervals or preserve a full-day state trajectory. |
| `-r`, `--request` | `100` | Positive integer controlling arrival volume. Values `1`-`5` select predefined levels; larger values are literal multipliers. See the mapping below. |
| `-d`, `--duration` | `1` | Positive integer multiplier for execution durations only. Does not scale CPU, memory, I/O, or other feature fields. Scaled durations must not exceed 900 seconds. |
| `-dl`, `--delay` | `20` | Integer multiplier for the inter-region round-trip-time matrix. At `20`, adjacent, two-hop, and three-hop delays are 3, 6, and 9 seconds. |
| `-s`, `--slo` | `100` | Allowed relocation percentage, from `0` to `100`, for each source region's requests per function. `0` means no cross-region relocation; `100` permits all requests to be relocated. This is not a utility-loss threshold. |
| `-l`, `--laxity` | `10` | Deadline-laxity metadata stored in run configuration/output names. The simulation uses the environment's fixed deadline-laxity value of `10`, independently of this argument. |
| `-n`, `--node` | `4` | Number of routing regions; the only accepted value is `4`. It does not set physical servers per region; those are scenario constants in the environment. |
| `-t`, `--time` | `180` | Positive search-budget seconds, not simulated workload duration or a hard process timeout. Loading, initial evaluations, worker startup, and the final iteration affect wall time. Fixed-routing methods do not perform a time-budgeted search. |
| `-st`, `--step` | `5` | Integer routing step, divided by 100 internally: `5` means a share of `0.05`, and `10` means `0.10`. Used by search methods, not fixed routing. |
| `-p`, `--population` | `70` | Positive maximum retained population for SOFA, NSGA3, SWFAGA, and SOFA ablations. Not a worker-process count. Kimchi, GreenCourier, CASA, and fixed routing do not use it as a population control. |
| `-ff`, `--fit_frequency` | `5` | Positive retraining interval in search iterations for SOFA and the MLS ablation after the early random-search phase. Not seconds; not used to control training in the other methods. |
| `-rs`, `--random_search` | `15` | Early random-search iteration count for SOFA and the MLS ablation (`iter_early` in experiment tables). Does not affect fixed routing, Kimchi, GreenCourier, or CASA. NSGA3 and SWFAGA accept the value through the shared interface but do not use it to control their current search loop. |
| `--casa-utility-target` | `2.0` | CASA-only utility-loss target, expressed in percentage points from `0` to `100`. CASA prioritizes utility improvement above this target and carbon improvement once it is met. This is separate from `--slo`. |
| `--seed`, `--casa-seed` | `0` | Shared random seed for all frameworks, Python/NumPy, and worker tasks. Integer from `0` to `4294967295`; both names set the same option. Time-budgeted runs may finish different iteration counts even with the same seed. |
| `--output-root` | `.` | Directory containing the per-framework, per-seed run folders. Each invocation gets a unique run directory, so previous runs are preserved. |

### Framework choices

Use the exact value in the first column with `--framework` or `-f`.

| Value | Method |
| --- | --- |
| `SOFA` | Full SOFA search, combining model-guided local search and evolutionary search. |
| `NSGA3` | NSGA-III comparison framework. |
| `SWFAGA` | SWFAGA comparison framework. |
| `kimchi` | Kimchi comparison framework. |
| `greenc` | GreenCourier comparison framework. |
| `CASA` | Utility-target-aware, utility/carbon search adaptation. |
| `SOFA_DEA` | Evolutionary-only SOFA ablation; no model-guided/local search. |
| `SOFA_MLS` | Model-guided local-search SOFA ablation; no evolutionary offspring. |
| `SOFA_RandomLS` | Randomly selected local-search SOFA ablation; no model guidance or evolutionary offspring. |
| `even` | Fixed routing shares `[0.25, 0.25, 0.25, 0.25]`. |
| `coast` | Fixed routing shares `[0.4, 0.1, 0.1, 0.4]`. |
| `centre` | Fixed routing shares `[0.1, 0.4, 0.4, 0.1]`. |
| `change` | Route according to the selected interval's original request-origin shares. |

### Request-origin choices

`--origin` describes the incoming workload; `--framework` chooses its routing.
For example, `-o coast -f even` routes coast-originating requests evenly.
Shares below follow the environment's region order.

| Value | Origin shares |
| --- | --- |
| `even` | `[0.25, 0.25, 0.25, 0.25]` |
| `coast` | `[0.4, 0.1, 0.1, 0.4]` |
| `centre` | `[0.1, 0.4, 0.4, 0.1]` |
| `change` | Interval-dependent regional shift. Starts at `[0.7, 0.1, 0.1, 0.1]` in interval 1 and is `[0.1, 0.1, 0.11875, 0.68125]` in interval 96. |

### Request-volume mapping

The multiplier is applied to the workload's arrival counts.

| `--request` | Effective multiplier |
| --- | ---: |
| `1` | 3330 |
| `2` | 6650 |
| `3` | 9980 |
| `4` | 13300 |
| `5` | 16630 |
| Any integer greater than `5` | The supplied integer itself |

For example, `-r 6` means multiply by 6, not a sixth workload level. The default
`-r 100` means multiply by 100. Use `-r 3` to select the third predefined level.

### Search-budget limitations

The multiprocessing search methods currently start 12 worker processes;
there is no command-line worker-count option. On Windows, startup and workload
transfer can take longer than a short search budget. A run can therefore
complete with initial policies but no search iterations. Use a sufficiently
long budget for optimization experiments. The 900-second example is a search
budget, not a promise that the process finishes in exactly 15 minutes.

## Summarize multiple seed runs

[summarize_seed_runs.py](summarize_seed_runs.py) reads completed run summaries,
groups matching experiment configurations, and writes `per_run.csv`,
`summary.csv`, and `summary.json` under `seed_comparison/`. The default batch
expects one completed run for each of the six main frameworks and seeds 1-10.
Duplicate runs for the same framework, seed, and configuration are rejected.

For example, summarize two completed SOFA seeds with a shared PHV reference:

```sh
python summarize_seed_runs.py results --frameworks SOFA --seeds 1 2 --reference 20 1000 1500 15000
```

The reference bounds in this example are illustrative. Choose justified bounds
and keep them fixed across experiments being compared.

| Argument or option | Default | Meaning |
| --- | --- | --- |
| `root` | Required | Directory to search recursively for `run_summary.json` files, such as `results`. |
| `-h`, `--help` | No | Show summary-command help and exit. |
| `--seeds` | `1 2 3 4 5 6 7 8 9 10` | Space-separated seed IDs to include. IDs must be unique. |
| `--frameworks` | `SOFA NSGA3 SWFAGA kimchi greenc CASA` | Space-separated methods to include. Also accepts `SOFA_DEA`, `SOFA_MLS`, and `SOFA_RandomLS`; fixed-routing names are not accepted by this command. |
| `--reference` | Automatic | Four positive finite raw-unit bounds, ordered as utility loss (%), cost (USD), carbon (kg), and water (L). Without this option, utility loss is bounded at 20 and the other bounds are 110% of pooled final-objective maxima, with a small positive floor. PHV is normalized by this reference. |
| `--confidence` | `0.95` | Confidence level strictly between 0 and 1 for a two-sided Student-t interval on the mean across seeds. Min/max are observed extremes, not confidence limits. |
| `--allow-partial` | Off | Allow missing requested framework/seed pairs for diagnostic reports. Does not allow duplicates or invalid objectives. Actual sample counts are reported. |

Automatically selected reference bounds can differ between batches. Do not
compare their PHV magnitudes unless the reference is identical. See
[PHV conventions](docs/PHV.md) for the distinction between normalized summary
PHV and the simulator's raw-unit PHV.

## Tests

```sh
python -m unittest discover -s . -p "test_*.py"
```

## More information

### Repository documentation

- [Workload datasets](docs/WORKLOADS.md): 845 functions across 96 intervals.
- [Objective units and PHV conventions](docs/PHV.md).
- [Simulation environment](ENV_V3.py).
- Run `python 15min_simulator.py --help` for all options.

### External datasets and environmental inputs

The bundled `.pkl.gz` files are processed simulation inputs, not complete copies
of the upstream databases. Regional solar, price, carbon, and water parameters
are stored as scenario constants in [ENV_V3.py](ENV_V3.py); the simulator does
not fetch live data. Current upstream values may differ from the bundled
inputs. These links do not specify the exact dates, locations, or preprocessing
used to construct every bundled value.

| Input | Source and URL | Purpose |
| --- | --- | --- |
| Huawei production FaaS workloads | [Huawei / SIR Lab data release](https://github.com/sir-lab/data-release), including the [2023 release documentation](https://github.com/sir-lab/data-release/blob/main/README_data_release_2023.md) | Compute-heavy trace source. |
| Azure workload repository | [Microsoft Azure Public Dataset](https://github.com/Azure/AzurePublicDataset) | Main repository for the Azure traces. |
| Azure I/O-heavy workload | [Azure Functions Blob Access Trace 2020](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsBlobDataset2020.md) | Trace associated with *Faa$T: A Transparent Auto-Scaling Cache for Serverless Applications*. |
| Azure workload characterization | [Azure Functions Trace 2019](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md) | Additional workload/latency characterization; not interchangeable with the Blob Access trace. |
| Solar radiation / DHI | [National Solar Radiation Database (NSRDB)](https://nsrdb.nlr.gov/), [data viewer](https://nsrdb.nlr.gov/data-viewer), [dataset information](https://www.nlr.gov/hpc/nsrdb-dataset), and [download API documentation](https://developer.nlr.gov/docs/solar/nsrdb/) | Diffuse horizontal irradiance for the on-site solar model. |
| Electricity carbon factors | [U.S. EIA: CO2 emissions per kWh](https://www.eia.gov/tools/faqs/faq.php?id=74&t=11) | Electricity carbon-intensity information; not an hourly city-level feed. |
| Regional carbon-data navigation | [EIA State Electricity Profiles](https://www.eia.gov/electricity/state/) | State-level generation/emissions data; not a record of the exact simulator constants. |
| Electricity-generation water intensity | [Consumptive Water Use for U.S. Power Production](https://docs.nlr.gov/docs/fy04osti/33905.pdf), Torcellini, Long, and Judkoff, NREL/TP-550-33905 | Energy-water intensity factor (EWIF) source. |
| On-site and source water accounting | [The Green Grid: Water Usage Effectiveness (WUE), White Paper 35](https://www.thegreengrid.org/system/files/store/WUE_v1.pdf) | Water-accounting and cooling-water background. |
| Potable-water and wastewater energy intensity | [Malinowski et al., Energy-Water Nexus](https://doi.org/10.1061/%28ASCE%29WR.1943-5452.0000528) | Literature source for water-production/treatment energy intensities. The current release's carbon objective includes grid electricity only. |

NSRDB provides DHI as one of its irradiance variables; DHI is not a separate
database. For access methods and any required credentials, consult the source
provider. The original sources retain their own terms of use.

