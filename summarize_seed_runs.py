"""Summarize independent seed runs, not individual Pareto policies, as samples."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t
import hvwfg

FRAMEWORKS = ('SOFA', 'NSGA3', 'SWFAGA', 'kimchi', 'greenc', 'CASA')
METRICS = ('utility_loss_percent', 'cost_USD', 'carbon_kg', 'water_L', 'phv')


def statistics(values, confidence=.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf((1+confidence)/2, n-1)*std/np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, min=float(values.min()), max=float(values.max()),
                std=std, ci_low=mean-half if half is not None else None,
                ci_high=mean+half if half is not None else None)


def hypervolume(values, reference):
    valid = values[(values < reference).all(axis=1)]
    return float(hvwfg.wfg(np.ascontiguousarray(valid/reference), np.ones(4))) if len(valid) else 0.


def experiment_configuration(record):
    return {k: v for k, v in record['configuration'].items()
            if k not in ('framework', 'seed', 'output_root')}


def summarize(root, expected_seeds, reference=None, confidence=.95, allow_partial=False,
              configuration_filter=None, frameworks=FRAMEWORKS):
    found, signature, all_values = {}, None, []
    for path in sorted(root.rglob('run_summary.json')):
        record = json.loads(path.read_text())
        name, seed = record['framework'], record['seed']
        if name not in frameworks or seed not in expected_seeds:
            continue
        config = experiment_configuration(record)
        if configuration_filter is not None and config != configuration_filter:
            continue
        key = (name, seed)
        if key in found:
            raise ValueError(f'Duplicate framework/seed {key}; select one experiment directory')
        if record['status'] != 'completed':
            continue
        if signature is not None and config != signature:
            raise ValueError(f'Mixed configurations: {path}')
        signature = config
        values = np.asarray(record['objectives'], dtype=float)
        if values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f'Invalid objective matrix: {path}')
        if (values < 0).any() or (values[:, 0] > 100).any():
            raise ValueError(f'Out-of-range objectives: {path}')
        found[key] = (values, path)
        all_values.append(values)
    missing = [f'{name}/seed{seed}' for name in frameworks for seed in expected_seeds if (name, seed) not in found]
    if missing and not allow_partial:
        raise ValueError('Missing runs: '+', '.join(missing))
    if not all_values:
        raise ValueError('No completed seed runs found')
    if reference is None:
        reference = np.maximum(np.vstack(all_values).max(axis=0)*1.1, 1e-12)
        reference[0] = 20.
        method = '20% utility; other coordinates 110% of pooled final-objective maxima'
    else:
        reference = np.asarray(reference, dtype=float)
        method = 'User-specified fixed reference'
    if reference.shape != (4,) or not np.isfinite(reference).all() or (reference <= 0).any():
        raise ValueError('Reference must contain four positive finite values')
    per_run, rows = [], []
    for (name, seed), (values, path) in found.items():
        metrics = list(values.min(axis=0)) + [hypervolume(values, reference)]
        per_run.append(dict(framework=name, seed=seed, source=str(path),
                            **dict(zip(METRICS, map(float, metrics)))))
    for name in frameworks:
        group = [r for r in per_run if r['framework'] == name]
        if not group:
            continue
        for metric in METRICS:
            rows.append(dict(framework=name, metric=metric,
                             **statistics([r[metric] for r in group], confidence)))
    return dict(confidence=confidence, ci_method='Two-sided Student t interval for the mean across independent seeds',
                notes=['Min/max are observed extremes, not confidence limits.',
                       'Each seed contributes one minimum per objective; minima can belong to different policies.',
                       'PHV is computed on final returned objectives with one shared normalized reference box.',
                       'Wall-clock budgets may produce different iteration counts even for the same seed.',
                       'Intervals describe run-to-run variability, not uncertainty in the physical model.'],
                reference=reference.tolist(), reference_method=method, configuration=signature,
                expected_seeds=expected_seeds, missing=missing, per_run=per_run, statistics=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    parser.add_argument('--frameworks', nargs='+', default=list(FRAMEWORKS),
                        choices=list(FRAMEWORKS)+['SOFA_DEA','SOFA_MLS','SOFA_RandomLS'])
    parser.add_argument('--reference', type=float, nargs=4, metavar=('UTILITY','COST','CARBON','WATER'))
    parser.add_argument('--confidence', type=float, default=.95)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    if not 0 < args.confidence < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('confidence must be in (0,1); seed IDs must be unique')
    configurations = {}
    for path in sorted(args.root.rglob('run_summary.json')):
        record = json.loads(path.read_text())
        if record['framework'] not in args.frameworks or record['seed'] not in args.seeds:
            continue
        config = experiment_configuration(record)
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        configurations[key] = config
    try:
        if not configurations:
            raise ValueError('No matching seed runs found')
        reports = {key: summarize(args.root, args.seeds, args.reference, args.confidence,
                                  args.allow_partial, config, args.frameworks)
                   for key, config in configurations.items()}
    except ValueError as error:
        parser.error(str(error))
    out = args.root/'seed_comparison'
    out.mkdir(exist_ok=True)
    (out/'experiments.json').write_text(json.dumps(configurations, indent=2))
    for key, report in reports.items():
        destination = out if len(reports) == 1 else out/key
        destination.mkdir(exist_ok=True)
        (destination/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        for filename, rows in [('summary.csv',report['statistics']),('per_run.csv',report['per_run'])]:
            with (destination/filename).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print('Experiment:', key)
        for r in report['statistics']:
            print(f"{r['framework']:7} {r['metric']:22} n={r['n']} mean={r['mean']:.6g} min={r['min']:.6g} max={r['max']:.6g} CI=[{r['ci_low']}, {r['ci_high']}]")
    print('Saved:', out)


if __name__ == '__main__':
    main()
