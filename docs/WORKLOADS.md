# Workload inputs

Run the simulator from the repository directory. The synthetic workload pair
is bundled as losslessly compressed gzip files. The simulator accepts either
the plain filenames below or their `.gz` equivalents; no extraction step is
needed.

| `--workload` | Arrival file | Feature lookup file | Function count |
| --- | --- | --- | ---: |
| `synthetic` | `synthetic_workload_arrival_pattern_arr.pkl` | `synthetic_workload_feature_lut_list.pkl` | 845 |
| `compute` | `R5_compute_workload_arrival_pattern_arr.pkl` | `R5_compute_workload_feature_lut_list.pkl` | 487 |
| `io` | `io_workload_arrival_pattern_arr.pkl` | `io_workload_feature_lut_list.pkl` | 358 |

Only the `synthetic` pair is bundled. The `compute` and `io` workload modes
require supplying their matching files separately.

The bundled files are:

- `synthetic_workload_arrival_pattern_arr.pkl.gz`
- `synthetic_workload_feature_lut_list.pkl.gz`

The bundled workload covers 96 intervals and 845 functions. Each interval's
arrival matrix has shape `(845, 16)`: a total plus 15 one-minute counts. The
compressed pair occupies about 8.5 MB instead of about 112 MB uncompressed.

[DATASETS.json](../DATASETS.json) records compressed and original byte counts
and SHA-256 hashes. Decompressing each bundled file reproduces its source file
byte for byte. File compression does not change the simulation inputs.

Arrivals are indexed by interval, function, and time bin. The first column is
the request total; the remaining columns describe within-interval arrivals.
The selected interval must exist. Feature lookup rows contain five fields;
field zero is a nonnegative integer execution duration in seconds, and scaled
execution durations cannot exceed the 900-second resource buffer. Consult
the environment's feature unpacking for the remaining resource/utility fields;
do not substitute an unrelated trace based only on matching array dimensions.

Python pickle can execute code when loading. Load only trusted workload files;
this release does not download or automatically unpickle external datasets.
