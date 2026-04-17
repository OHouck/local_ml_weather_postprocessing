#!/usr/bin/env python3
"""
Run architecture experiments on a 10% eval sample of continent 6x6 cells.

This script:
1. Samples a deterministic 10% "eval" subset of continent cells (disjoint from hyperopt subset)
2. Runs each experiment configuration on each cell
3. Saves output zarrs with continent/patch info so plot_arch_experiment_results.py can aggregate

Usage:
    python3 finetuning/run_arch_experiments_eval.py
"""

import os
import sys
import numpy as np
import torch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories, generate_output_path, sample_continent_patches
from finetuning.finetune import (
    run_subregion_experiment, load_optimal_hyperparameters, parse_args
)

USE_LEGACY_GLOBAL_DATA = True # can be true if only using 2m temp and 10m wind speed


def make_base_args(data_dir, output_dir, model_name, train_start, train_end,
                   test_start, test_end, lead_time_hours):
    """Create base args namespace with common settings."""
    return SimpleNamespace(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        subregion="6x6",
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        training_vars=["2m_temperature"],
        output_vars=["2m_temperature"],
        lead_time_hours=lead_time_hours,
        alternate_loss_fn=None,
        growing_season_only=False,
        bootstrap=None,
        ground_truth_source="",
        pca_components=0,
        # MLP defaults (will be overridden by optimal hyperparams)
        mlp_hidden_dim=1024,
        mlp_num_layers=6,
        mlp_dropout=0.25,
        unet_hidden_dim=64,
        unet_dropout=0.1,
        optimal_lr=None,
        optimal_batch_size=None,
        optimal_weight_decay=None,
        optimal_patience=None,
        optimal_min_delta=None,
        optimal_lead_time_embedding_dim=None,
        optimal_snapshot_T0=None,
        optimal_snapshot_T_mult=None,
        # New features defaults
        lead_time_loss_weights=None,
        cmixup_alpha=0.0,
        mc_dropout_samples=0,
        per_lead_time=False,
        small_output_init=False,
    )


# ---------------------------------------------------------------------------
# Variable configurations to loop over.
# Each entry defines what the model trains on and what it predicts.
# Comment out any configs you don't want to run.
# ---------------------------------------------------------------------------
VAR_CONFIGS = [
    {
        'label': '2m Temperature',
        'training_vars': ['2m_temperature'],
        'output_vars':   ['2m_temperature'],
    },
    {
        'label': '10m Wind Speed',
        'training_vars': ['10m_wind_speed'],
        'output_vars':   ['10m_wind_speed'],
    },
    {
        'label': '2m Temperature + 1000hPa T & q',
        'training_vars': ['2m_temperature', 'temperature_1000hPa', 'specific_humidity_1000hPa'],
        'output_vars':   ['2m_temperature'],
    },
]

# ---------------------------------------------------------------------------
# Architecture templates — one entry per model variant.
# training_vars / output_vars are filled in from VAR_CONFIGS above.
# ---------------------------------------------------------------------------
_ARCH_TEMPLATES = [
    {
        'name_prefix': 'MLP',
        'nn_architecture': 'mlp',
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': None,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name_prefix': 'MLP Snapshot Ensemble x3',
        'nn_architecture': 'mlp',
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': 3,
        'snapshot_epochs': 210,
        'snapshot_T0': 30,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name_prefix': 'UNet',
        'nn_architecture': 'unet',
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': None,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name_prefix': 'Block LTHO Ensemble',
        'nn_architecture': 'mlp',
        'block_ensemble': True,
        'block_holdout': 3,
        'snapshot_ensemble': 1,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    # Experiment 1: Per-Lead-Time MLP Snapshot x3
    # Trains a separate snapshot ensemble for each lead time so each model optimizes
    # purely for its horizon without gradient competition from other lead times.
    {
        'name_prefix': 'Per-LT MLP Snapshot x3',
        'nn_architecture': 'mlp',
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': 3,
        'snapshot_epochs': 210,
        'snapshot_T0': 30,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
        'per_lead_time': True,
    },
]

# Build the full experiment list by crossing each architecture template with
# each variable configuration.
EXPERIMENTS = []
for var_cfg in VAR_CONFIGS:
    for tmpl in _ARCH_TEMPLATES:
        exp = dict(tmpl)
        exp['name'] = f"{tmpl['name_prefix']} ({var_cfg['label']})"
        exp['training_vars'] = var_cfg['training_vars']
        exp['output_vars']   = var_cfg['output_vars']
        exp.pop('name_prefix')
        EXPERIMENTS.append(exp)


def main():
    dirs = setup_directories()
    data_dir = dirs['raw']
    output_dir = dirs['input']

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    # Sample the eval 5% of continent cells
    eval_cells = sample_continent_patches(
        dirs['processed'], fraction=0.05, seed=42, split='eval'
    )
    print(f"\nLoaded {len(eval_cells)} eval cells from continent 10% sample")

    # Configuration
    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2022-12-31"
    model_name = "pangu"
    lead_time_hours = [24, 120, 216]

    # Pre-load hyperparameter sets so we don't re-read JSON on every cell.
    # Each experiment picks the set that matches its architecture / training mode.
    _hp_cache = {}

    def get_hyperparams(arch, training_vars, use_snapshot, use_block_ltho):
        key = (arch, tuple(training_vars), use_snapshot, use_block_ltho)
        if key not in _hp_cache:
            _hp_cache[key] = load_optimal_hyperparameters(
                arch, training_vars, ['2m_temperature'],
                alternate_loss_fn=None,
                use_snapshot=use_snapshot,
                use_block_ltho=use_block_ltho,
            )
        return _hp_cache[key]

    for exp in EXPERIMENTS:
        print(f"\n{'#'*70}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"{'#'*70}")

        if '10m_wind_speed' in exp['training_vars'] or 'temperature_1000hPa' in exp['training_vars']:
            print("  Skipping non 2m temp experiments")
            continue

        for i, (continent, patch_idx, patch_array) in enumerate(eval_cells):
            lat_vals = patch_array[0]
            lon_vals = patch_array[1]

            print(f"\n{'='*70}")
            print(f"[{exp['name']}] Cell {i+1}/{len(eval_cells)}: {continent} patch {patch_idx}")
            print(f"  Lat: {lat_vals.min():.2f} to {lat_vals.max():.2f}")
            print(f"  Lon: {lon_vals.min():.2f} to {lon_vals.max():.2f}")
            print(f"{'='*70}")

            # Build args
            args = make_base_args(data_dir, output_dir, model_name,
                                  train_start, train_end, test_start, test_end,
                                  lead_time_hours)
            args.region = continent
            args.training_vars = exp.get('training_vars', args.training_vars)
            args.output_vars = args.training_vars[:1]  # always correct only the first var
            args.nn_architecture = exp['nn_architecture']
            args.block_ensemble = exp['block_ensemble']
            args.block_holdout = exp['block_holdout']
            args.snapshot_ensemble = exp['snapshot_ensemble']
            args.snapshot_epochs = exp['snapshot_epochs']
            args.snapshot_T0 = exp['snapshot_T0']
            args.snapshot_T_mult = exp['snapshot_T_mult']
            args.ensemble = exp.get('ensemble')
            args.swa_ensemble = exp.get('swa_ensemble')
            args.lead_time_loss_weights = exp.get('lead_time_loss_weights')
            args.cmixup_alpha = exp.get('cmixup_alpha', 0.0)
            args.per_lead_time = exp.get('per_lead_time', False)
            args.small_output_init = exp.get('small_output_init', False)

            # Apply optimal hyperparameters — select the right set per experiment.
            # Block LTHO uses its own dedicated hyperopt; plain MLP/Snapshot use
            # the snapshot-tuned set; UNet uses non-snapshot results.
            use_block = exp['block_ensemble']
            use_snap = exp['snapshot_ensemble'] is not None
            hp = get_hyperparams(
                exp['nn_architecture'],
                exp.get('training_vars', ['2m_temperature']),
                use_snapshot=use_block or use_snap,
                use_block_ltho=use_block,
            )
            if hp:
                args.mlp_hidden_dim = hp.get('hidden_dim', args.mlp_hidden_dim)
                args.mlp_num_layers = hp.get('num_layers', args.mlp_num_layers)
                args.mlp_dropout = hp.get('dropout_rate', args.mlp_dropout)
                args.unet_hidden_dim = hp.get('hidden_dim', args.unet_hidden_dim)
                args.unet_dropout = hp.get('dropout_rate', args.unet_dropout)
                args.optimal_lr = hp.get('learning_rate')
                args.optimal_batch_size = hp.get('batch_size')
                args.optimal_weight_decay = hp.get('weight_decay')
                args.optimal_patience = hp.get('patience')
                args.optimal_min_delta = hp.get('min_delta')
                args.optimal_lead_time_embedding_dim = hp.get('lead_time_embedding_dim')
                args.optimal_snapshot_T0 = hp.get('snapshot_T0')
                args.optimal_snapshot_T_mult = hp.get('snapshot_T_mult')

            # Generate output path with eval cell identifier
            base_path = os.path.join(output_dir, generate_output_path(args))
            out_path = base_path.replace('.zarr', f'_{continent}_bs{patch_idx}.zarr')

            print(f"  Output: {out_path}")

            # Skip if already exists
            if os.path.exists(out_path):
                print(f"  Skipping (already exists)")
                continue

            try:
                run_subregion_experiment(
                    lat_vals, lon_vals, out_path,
                    args, data_dir, device, patch_num=patch_idx,
                    use_legacy_global_data=USE_LEGACY_GLOBAL_DATA
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # --- Pooled FiLM experiments (train once globally across all eval cells) ---
    pooled_exps = [e for e in EXPERIMENTS if e.get('nn_architecture') == 'pooled_film']
    for exp in pooled_exps:
        print(f"\n{'#'*70}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"{'#'*70}")
        run_pooled_film_experiment(
            exp, eval_cells, dirs, device,
            train_start, train_end, test_start, test_end,
            model_name, lead_time_hours
        )

    print(f"\n{'='*70}")
    print("All experiments complete.")
    print(f"{'='*70}")


def run_pooled_film_experiment(exp, eval_cells, dirs, device, train_start, train_end,
                               test_start, test_end, model_name, lead_time_hours):
    """
    Train a single PooledFiLMMLP across all eval cells and save per-patch zarr outputs.

    Instead of training one model per patch, loads all patch data, concatenates it with
    region descriptors, trains once globally, then runs per-patch inference.

    Args:
        exp: Experiment config dict from _ARCH_TEMPLATES
        eval_cells: List of (continent, patch_idx, patch_array) from sample_continent_patches
        dirs: Directory dict from setup_directories()
        device: Torch device
        train_start/end, test_start/end: Date range strings
        model_name: Forecast model name ('pangu')
        lead_time_hours: List of lead time hours
    """
    import math
    from finetuning.finetune import (
        load_optimal_hyperparameters, PooledFiLMMLP, train_pooled_film_model
    )
    from finetuning.prepare_forecasts_and_targets import load_forecasts
    from finetuning.finetune import save_output
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np

    data_dir = dirs['raw']
    output_dir = dirs['input']

    args_template = make_base_args(
        data_dir, output_dir, model_name, train_start, train_end,
        test_start, test_end, lead_time_hours
    )
    args_template.training_vars = exp.get('training_vars', ['2m_temperature'])
    args_template.output_vars = exp.get('output_vars', ['2m_temperature'])
    args_template.nn_architecture = 'pooled_film'
    args_template.block_ensemble = exp['block_ensemble']
    args_template.block_holdout = exp['block_holdout']
    args_template.snapshot_ensemble = exp['snapshot_ensemble']
    args_template.snapshot_epochs = exp['snapshot_epochs']
    args_template.snapshot_T0 = exp['snapshot_T0']
    args_template.snapshot_T_mult = exp['snapshot_T_mult']
    args_template.ensemble = exp.get('ensemble')
    args_template.swa_ensemble = exp.get('swa_ensemble')

    hp = load_optimal_hyperparameters(
        'mlp', args_template.training_vars, ['2m_temperature'],
        alternate_loss_fn=None, use_snapshot=True, use_block_ltho=False
    )

    snapshot_epochs = exp.get('snapshot_epochs', 270)
    snapshot_T0 = exp.get('snapshot_T0', 90)
    snapshot_T_mult = exp.get('snapshot_T_mult', 1)
    lr = hp.get('learning_rate', 3.3e-4) if hp else 3.3e-4
    weight_decay = hp.get('weight_decay', 2.2e-6) if hp else 2.2e-6
    batch_size = hp.get('batch_size', 128) if hp else 128
    lead_time_emb_dim = hp.get('lead_time_embedding_dim', 4) if hp else 4

    print(f"\n{'='*70}")
    print(f"POOLED FiLM EXPERIMENT: {exp.get('name', 'Pooled FiLM')}")
    print(f"  Loading data from {len(eval_cells)} eval cells...")
    print(f"{'='*70}")

    # Region descriptor: [sin(lat), cos(lat), sin(lon), cos(lon)]
    region_dim = 4
    n_lead_times = len(lead_time_hours)

    # --- Load data from all patches ---
    all_fc, all_fc_out, all_obs = [], [], []
    all_lti, all_doy, all_region = [], [], []
    patch_test_data = []  # per-patch test data for inference

    for continent, patch_idx, patch_array in eval_cells:
        lat_vals = patch_array[0]
        lon_vals = patch_array[1]

        args_patch = make_base_args(
            data_dir, output_dir, model_name, train_start, train_end,
            test_start, test_end, lead_time_hours
        )
        args_patch.training_vars = args_template.training_vars
        args_patch.output_vars = args_template.output_vars
        args_patch.nn_architecture = 'pooled_film'
        args_patch.region = continent
        args_patch.block_ensemble = False
        args_patch.block_holdout = 1
        args_patch.snapshot_ensemble = None
        args_patch.ensemble = None
        args_patch.swa_ensemble = None

        try:
            (fc, fc_output, obs, lti, doy, train_times, lat_u, lon_u,
             n_lat, n_lon, n_training_vars, n_output_vars, tmfe) = load_forecasts(
                data_dir, args_patch, lat_vals, lon_vals, train=True,
                patch_num=patch_idx, use_legacy_global_data=USE_LEGACY_GLOBAL_DATA
            )
            (test_fc, test_fc_out, test_obs, test_lti, test_doy, test_times,
             _, _, _, _, _, _, _) = load_forecasts(
                data_dir, args_patch, lat_vals, lon_vals, train=False,
                patch_num=patch_idx, use_legacy_global_data=USE_LEGACY_GLOBAL_DATA
            )
        except Exception as e:
            print(f"  Skipping {continent} patch {patch_idx}: {e}")
            continue

        stats_in = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
        stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}
        fc_norm = (fc - stats_in['mean']) / stats_in['std']
        fc_out_norm = (fc_output - stats_out['mean']) / stats_out['std']
        obs_norm = (obs - stats_out['mean']) / stats_out['std']

        # Region descriptor from patch center
        lat_center = float(lat_vals.mean())
        lon_center = float(lon_vals.mean())
        region_desc = np.array([
            math.sin(math.radians(lat_center)),
            math.cos(math.radians(lat_center)),
            math.sin(math.radians(lon_center)),
            math.cos(math.radians(lon_center)),
        ], dtype=np.float32)
        region_tile = np.tile(region_desc, (len(fc_norm), 1))

        all_fc.append(fc_norm.astype(np.float32))
        all_fc_out.append(fc_out_norm.astype(np.float32))
        all_obs.append(obs_norm.astype(np.float32))
        all_lti.append(lti)
        all_doy.append(doy.astype(np.float32))
        all_region.append(region_tile)

        test_fc_norm = (test_fc - stats_in['mean']) / stats_in['std']
        test_fc_out_norm = (test_fc_out - stats_out['mean']) / stats_out['std']
        patch_test_data.append({
            'continent': continent, 'patch_idx': patch_idx,
            'lat_vals': lat_vals, 'lon_vals': lon_vals,
            'test_fc_norm': test_fc_norm.astype(np.float32),
            'test_fc_out_norm': test_fc_out_norm.astype(np.float32),
            'test_obs': test_obs,
            'test_lti': test_lti,
            'test_doy': test_doy.astype(np.float32),
            'test_times': test_times,
            'stats_out': stats_out,
            'region_desc': region_desc,
            'tmfe': tmfe,
            'n_lat': n_lat, 'n_lon': n_lon,
            'args_patch': args_patch,
        })
        print(f"  Loaded {continent} patch {patch_idx}: {len(fc_norm)} train samples")

    if not all_fc:
        print("  No patches loaded — skipping pooled FiLM.")
        return

    # Concatenate all patches
    fc_pool = np.concatenate(all_fc, axis=0)
    fc_out_pool = np.concatenate(all_fc_out, axis=0)
    obs_pool = np.concatenate(all_obs, axis=0)
    lti_pool = np.concatenate(all_lti, axis=0)
    doy_pool = np.concatenate(all_doy, axis=0)
    region_pool = np.concatenate(all_region, axis=0)

    print(f"\n  Pooled dataset: {fc_pool.shape[0]} samples from {len(patch_test_data)} patches")

    # Train/val split — use a local RNG to avoid mutating global numpy state
    n_total = len(fc_pool)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_total)
    split = int(0.8 * n_total)
    t_idx, v_idx = idx[:split], idx[split:]

    def make_tensor_dataset(i):
        return TensorDataset(
            torch.from_numpy(fc_pool[i]),
            torch.from_numpy(fc_out_pool[i]),
            torch.from_numpy(obs_pool[i]),
            torch.from_numpy(lti_pool[i].astype(np.int64)),
            torch.from_numpy(doy_pool[i]),
            torch.from_numpy(region_pool[i]),
        )

    train_ds = make_tensor_dataset(t_idx)
    val_ds = make_tensor_dataset(v_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = fc_pool.shape[1]
    output_dim = fc_out_pool.shape[1]

    model = PooledFiLMMLP(
        input_dim=input_dim, region_dim=region_dim, output_dim=output_dim,
        hidden_dim=256, num_layers=4, n_lead_times=n_lead_times,
        lead_time_embedding_dim=lead_time_emb_dim, dropout_rate=0.25
    ).to(device)

    snapshots, training_time = train_pooled_film_model(
        model, train_loader, val_loader,
        epochs=snapshot_epochs, lr=lr, device=device,
        weight_decay=weight_decay, grad_clip=1.0,
        T_0=snapshot_T0, T_mult=snapshot_T_mult,
    )

    if not snapshots:
        print("  No snapshots collected — check snapshot_T0 vs epochs.")
        return

    # --- Per-patch inference ---
    for patch_info in patch_test_data:
        continent = patch_info['continent']
        patch_idx = patch_info['patch_idx']

        base_path = os.path.join(output_dir, generate_output_path(patch_info['args_patch']))
        out_path = base_path.replace('.zarr', f'_{continent}_bs{patch_idx}.zarr')

        if os.path.exists(out_path):
            print(f"  Skipping {continent} patch {patch_idx} (already exists)")
            continue

        test_fc_norm = patch_info['test_fc_norm']
        test_fc_out_norm = patch_info['test_fc_out_norm']
        test_lti = patch_info['test_lti']
        test_doy = patch_info['test_doy']
        stats_out = patch_info['stats_out']
        region_desc = patch_info['region_desc']

        region_tensor = torch.from_numpy(
            np.tile(region_desc, (len(test_fc_norm), 1))
        ).float().to(device)

        snap_corrections = []
        for snap_weights, _ in snapshots:
            model.load_state_dict(snap_weights)
            model.eval()
            all_corr = []
            bs = 128
            with torch.no_grad():
                for i in range(0, len(test_fc_norm), bs):
                    end = min(i + bs, len(test_fc_norm))
                    fc_in = torch.from_numpy(test_fc_norm[i:end]).float().to(device)
                    fc_out = torch.from_numpy(test_fc_out_norm[i:end]).float().to(device)
                    lt = torch.from_numpy(test_lti[i:end].astype(np.int64)).to(device)
                    doy = torch.from_numpy(test_doy[i:end]).float().to(device)
                    reg = region_tensor[i:end]
                    pred_err = model(fc_in, reg, lt, doy)
                    corr = (fc_out + pred_err).cpu().numpy()
                    all_corr.append(corr)
            corr_norm = np.concatenate(all_corr, axis=0)
            corr_phys = (corr_norm * stats_out['std']) + stats_out['mean']
            snap_corrections.append(corr_phys)

        corrected = np.mean(snap_corrections, axis=0)
        original_phys = (patch_info['test_fc_out_norm'] * stats_out['std']) + stats_out['mean']

        print(f"  Saving {continent} patch {patch_idx} → {out_path}")
        try:
            save_output(
                out_path, model_name, patch_info['args_patch'].output_vars,
                patch_info['lon_vals'], patch_info['lat_vals'],
                patch_info['test_times'], lead_time_hours,
                original_phys, corrected, test_lti,
                ground_truth_data=patch_info['test_obs'],
                training_mean_forecast_error=patch_info['tmfe'],
                training_time_minutes=training_time,
            )
        except Exception as e:
            print(f"  ERROR saving {continent} patch {patch_idx}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n  Pooled FiLM complete. Total training time: {training_time:.2f} min")


if __name__ == "__main__":
    main()
