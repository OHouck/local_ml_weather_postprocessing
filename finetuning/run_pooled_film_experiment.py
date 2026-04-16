#!/usr/bin/env python3
"""
Pooled multi-region MLP with FiLM (Feature-wise Linear Modulation) conditioning.

Instead of training one independent model per 6x6 patch, this trains a SINGLE shared
MLP on data pooled across all continent eval patches, conditioned on a per-patch
region descriptor via FiLM. This effectively multiplies the training set by the number
of patches, directly addressing the small-per-region data limitation.

Source ideas:
  - Pooled station embedding: Rasp & Lerch 2018, MWR 146(11)
  - FiLM conditioning: Perez et al. 2018, AAAI
  - Multi-region validation: Schulz & Lerch 2022, MWR 150(1)

Usage:
    python3 finetuning/run_pooled_film_experiment.py

Output zarrs are written alongside the per-patch experiment outputs and named with
architecture tag 'pooled_film', so process_forecasts.py and plot_arch_experiment_results.py
pick them up automatically.
"""

import os
import sys
import copy
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories, generate_output_path, sample_continent_patches
from finetuning.finetune import (
    load_optimal_hyperparameters, create_dataloader, apply_correction, save_output
)
from finetuning.prepare_forecasts_and_targets import load_forecasts


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PooledFiLMMLP(nn.Module):
    """
    Shared MLP with FiLM region conditioning.

    A single backbone MLP trained on pooled data from many 6x6 patches. Each hidden
    layer is modulated by Feature-wise Linear Modulation (FiLM) conditioned on a
    region descriptor [sin(lat), cos(lat), sin(lon), cos(lon), elevation_mean, SDOR,
    koppen_zone_onehot (5)]. This lets the shared backbone adapt its activations
    per-region without needing separate model weights.

    The (1 + gamma) * h + beta FiLM formulation initialises as an identity modulation
    (gamma=0, beta=0) so that training begins as a standard shared MLP.

    Reference: Perez et al. 2018, AAAI "FiLM: Visual Reasoning with a General
    Conditioning Layer". Applied to pooled post-processing following Rasp & Lerch 2018.
    """

    # Region descriptor dimension:
    # sin(lat), cos(lat), sin(lon), cos(lon) = 4
    # elevation_mean, SDOR = 2
    # Koppen zone one-hot (5 zones: tropical, arid, temperate, cold, polar) = 5
    REGION_DIM = 11

    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4,
                 n_lead_times=1, lead_time_embedding_dim=8, dropout_rate=0.25):
        super().__init__()

        self.n_lead_times = n_lead_times
        self.lead_time_embedding = None
        self.hidden_dim = hidden_dim

        # Day-of-year sin/cos + optional lead time embedding
        actual_input_dim = input_dim + 2
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)
            actual_input_dim += lead_time_embedding_dim

        # Shared backbone layers
        self.input_proj = nn.Linear(actual_input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # FiLM hypernetwork: maps region descriptor -> (gamma, beta) for each hidden layer
        # Apply FiLM after input_proj and after each hidden layer (num_layers total)
        self.film_layers = nn.ModuleList([
            nn.Linear(self.REGION_DIM, 2 * hidden_dim) for _ in range(num_layers)
        ])

        # Initialize FiLM layers near zero so they start as identity modulations
        for film in self.film_layers:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        # Initialize output layer near zero (small-correction prior)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, region_desc, lead_time_idx=None, day_of_year_features=None):
        """
        Args:
            x: Forecast input features (batch, input_dim)
            region_desc: Region descriptor (batch, REGION_DIM)
            lead_time_idx: Lead time indices (batch,)
            day_of_year_features: sin/cos DOY (batch, 2)
        Returns:
            Predicted error correction (batch, output_dim)
        """
        if day_of_year_features is not None:
            x = torch.cat([x, day_of_year_features], dim=-1)
        if self.lead_time_embedding is not None and lead_time_idx is not None:
            lead_emb = self.lead_time_embedding(lead_time_idx)
            x = torch.cat([x, lead_emb], dim=-1)

        # Input projection + first FiLM
        h = self.input_proj(x)
        gamma_beta = self.film_layers[0](region_desc)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = (1.0 + gamma) * h + beta
        h = F.relu(h)
        h = self.dropout(h)

        # Hidden layers with FiLM
        for i, layer in enumerate(self.hidden_layers):
            h = layer(h)
            gamma_beta = self.film_layers[i + 1](region_desc)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
            h = F.relu(h)
            h = self.dropout(h)

        return self.output_layer(h)


# ---------------------------------------------------------------------------
# Region descriptor builder
# ---------------------------------------------------------------------------

KOPPEN_ZONES = ['tropical', 'arid', 'temperate', 'cold', 'polar']


def build_region_descriptor(lat_vals, lon_vals, continent=None):
    """
    Build a fixed region descriptor vector for a 6x6 patch.

    Uses center lat/lon from the patch arrays, plus simplified elevation and
    SDOR proxies (zero when not available — the model still learns lat/lon signal).

    Args:
        lat_vals: Array of latitude values in the patch
        lon_vals: Array of longitude values in the patch
        continent: Continent name string (used for Koppen zone one-hot; optional)

    Returns:
        region_desc: np.ndarray of shape (REGION_DIM,)
    """
    lat_c = float(np.mean(lat_vals))
    lon_c = float(np.mean(lon_vals))

    # Convert longitude to radians (lon in [0,360] -> center around 180)
    lon_rad = math.radians(lon_c)
    lat_rad = math.radians(lat_c)

    desc = [
        math.sin(lat_rad), math.cos(lat_rad),
        math.sin(lon_rad), math.cos(lon_rad),
        0.0,  # elevation_mean placeholder
        0.0,  # SDOR placeholder
    ]

    # Koppen zone one-hot: rough assignment by continent
    koppen_onehot = [0.0] * 5
    koppen_map = {
        'africa': 0,         # tropical
        'asia': 2,           # temperate (broad default)
        'europe': 2,         # temperate
        'north_america': 2,  # temperate
        'south_america': 0,  # tropical
        'oceania': 1,        # arid
    }
    if continent is not None and continent in koppen_map:
        koppen_onehot[koppen_map[continent]] = 1.0
    else:
        koppen_onehot[2] = 1.0  # default temperate

    desc.extend(koppen_onehot)
    return np.array(desc, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pooled_film_snapshot(model, train_loader, valid_loader, epochs, lr, device,
                                weight_decay=0, grad_clip=1.0, T_0=30, T_mult=1):
    """
    Train PooledFiLMMLP with cosine annealing snapshots.

    DataLoader batches must yield (fc_input, fc_output, obs, lead_time, doy, region_desc).
    Returns list of (state_dict, val_loss) snapshots.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-6)

    snapshots = []
    best_val_loss = float('inf')
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for fc_in, fc_out, y, lt, doy, rdesc in train_loader:
            fc_in = fc_in.to(device)
            fc_out = fc_out.to(device)
            y = y.to(device)
            lt = lt.to(device)
            doy = doy.to(device)
            rdesc = rdesc.to(device)

            optimizer.zero_grad()
            pred_error = model(fc_in, rdesc, lt, doy)
            preds = fc_out + pred_error
            loss = criterion(preds, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * fc_in.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Snapshot at cycle boundaries
        is_cycle_end = (epoch % T_0 == 0) if T_mult == 1 else False
        if not is_cycle_end and T_mult != 1:
            cycle_sum = 0
            for c in range(20):
                cycle_sum += T_0 * (T_mult ** c)
                if epoch == cycle_sum:
                    is_cycle_end = True
                    break

        if is_cycle_end:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for fc_in, fc_out, y, lt, doy, rdesc in valid_loader:
                    fc_in, fc_out, y = fc_in.to(device), fc_out.to(device), y.to(device)
                    lt, doy, rdesc = lt.to(device), doy.to(device), rdesc.to(device)
                    pred_error = model(fc_in, rdesc, lt, doy)
                    preds = fc_out + pred_error
                    val_loss += criterion(preds, y).item() * fc_in.size(0)
            val_loss /= len(valid_loader.dataset)
            snapshots.append((copy.deepcopy(model.state_dict()), val_loss))
            print(f"  Snapshot epoch {epoch}/{epochs}: val_loss={val_loss:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        if epoch % 50 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}, lr={lr_now:.2e}")

    elapsed = (time.time() - train_start) / 60.0
    print(f"  Training complete: {len(snapshots)} snapshots in {elapsed:.2f} min")
    return snapshots, elapsed


# ---------------------------------------------------------------------------
# Pooled DataLoader
# ---------------------------------------------------------------------------

class PooledPatchDataset(torch.utils.data.Dataset):
    """Dataset that concatenates data from multiple patches with their region descriptors."""

    def __init__(self, fc_inputs, fc_outputs, obs_list, lead_time_idxs, doy_features,
                 region_descs, patch_weights=None):
        """
        Args:
            fc_inputs: list of (n_samples_i, input_dim) arrays, one per patch
            fc_outputs: list of (n_samples_i, output_dim) arrays
            obs_list: list of (n_samples_i, output_dim) arrays
            lead_time_idxs: list of (n_samples_i,) int arrays
            doy_features: list of (n_samples_i, 2) arrays
            region_descs: list of (REGION_DIM,) arrays, one per patch
            patch_weights: optional per-patch sampling weights for stratification
        """
        all_fc_in, all_fc_out, all_obs = [], [], []
        all_lt, all_doy, all_rdesc = [], [], []

        for i, (fc_in, fc_out, obs, lt, doy, rdesc) in enumerate(
                zip(fc_inputs, fc_outputs, obs_list, lead_time_idxs, doy_features, region_descs)):
            n = fc_in.shape[0]
            all_fc_in.append(fc_in)
            all_fc_out.append(fc_out)
            all_obs.append(obs)
            all_lt.append(lt)
            all_doy.append(doy)
            # Broadcast region descriptor to all samples of this patch
            all_rdesc.append(np.tile(rdesc[np.newaxis], (n, 1)))

        self.fc_in = torch.from_numpy(np.concatenate(all_fc_in)).float()
        self.fc_out = torch.from_numpy(np.concatenate(all_fc_out)).float()
        self.obs = torch.from_numpy(np.concatenate(all_obs)).float()
        self.lt = torch.from_numpy(np.concatenate(all_lt)).long()
        self.doy = torch.from_numpy(np.concatenate(all_doy)).float()
        self.rdesc = torch.from_numpy(np.concatenate(all_rdesc)).float()

    def __len__(self):
        return len(self.fc_in)

    def __getitem__(self, idx):
        return (self.fc_in[idx], self.fc_out[idx], self.obs[idx],
                self.lt[idx], self.doy[idx], self.rdesc[idx])


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def apply_film_correction(model, fc_input, fc_output, lead_time_idxs, doy_features,
                           region_desc, device, batch_size=128):
    """
    Run inference with a PooledFiLMMLP on a single patch.

    Args:
        region_desc: (REGION_DIM,) numpy array for this patch
    Returns:
        corrected: (n_samples, output_dim) numpy array
    """
    model.eval()
    n = fc_input.shape[0]
    rdesc_tile = np.tile(region_desc[np.newaxis], (batch_size, 1))  # pre-tile, trim at end

    corrected_all = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            bs = end - i
            fc_in_b = torch.from_numpy(fc_input[i:end]).float().to(device)
            fc_out_b = torch.from_numpy(fc_output[i:end]).float().to(device)
            lt_b = torch.from_numpy(lead_time_idxs[i:end]).long().to(device)
            doy_b = torch.from_numpy(doy_features[i:end]).float().to(device)
            rd_b = torch.from_numpy(rdesc_tile[:bs]).float().to(device)

            pred_error = model(fc_in_b, rd_b, lt_b, doy_b)
            corrected = (fc_out_b + pred_error).cpu().numpy()
            corrected_all.append(corrected)

    return np.concatenate(corrected_all, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

USE_LEGACY_GLOBAL_DATA = False


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

    # Load same eval cell set as run_arch_experiments_eval.py for fair comparison
    eval_cells = sample_continent_patches(
        dirs['processed'], fraction=0.05, seed=42, split='eval'
    )
    print(f"\nLoaded {len(eval_cells)} eval cells")

    # Configuration — matches run_arch_experiments_eval.py
    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2022-12-31"
    model_name = "pangu"
    lead_time_hours = [24, 120, 216]
    n_lead_times = len(lead_time_hours)

    # Load optimal MLP hyperparameters as starting point for backbone
    optimal_hp = load_optimal_hyperparameters(
        'mlp', ['2m_temperature'], ['2m_temperature'],
        alternate_loss_fn=None, use_snapshot=True, use_block_ltho=True
    )

    lr = optimal_hp.get('learning_rate', 3.3e-4) if optimal_hp else 3.3e-4
    weight_decay = optimal_hp.get('weight_decay', 2.2e-6) if optimal_hp else 2.2e-6
    batch_size = optimal_hp.get('batch_size', 128) if optimal_hp else 128
    lead_time_emb_dim = optimal_hp.get('lead_time_embedding_dim', 4) if optimal_hp else 4

    print(f"\nHyperparameters: lr={lr}, wd={weight_decay}, batch={batch_size}")

    # ---------------------------------------------------------------------------
    # Phase 1: Load training data for all eval patches and build pooled dataset
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 1: Loading training data for all patches")
    print(f"{'='*60}")

    base_args = SimpleNamespace(
        data_dir=data_dir,
        model_name=model_name,
        subregion="6x6",
        training_vars=["2m_temperature"],
        output_vars=["2m_temperature"],
        lead_time_hours=lead_time_hours,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        alternate_loss_fn=None,
        growing_season_only=False,
        bootstrap=None,
        ground_truth_source="",
        pca_components=0,
    )

    # Collect per-patch data
    patch_train_data = []  # list of dicts
    patch_test_data = []

    for i, (continent, patch_idx, patch_array) in enumerate(eval_cells):
        lat_vals = patch_array[0]
        lon_vals = patch_array[1]
        print(f"  Loading patch {i+1}/{len(eval_cells)}: {continent} patch {patch_idx} "
              f"lat=[{lat_vals.min():.1f},{lat_vals.max():.1f}] "
              f"lon=[{lon_vals.min():.1f},{lon_vals.max():.1f}]")

        args = SimpleNamespace(**vars(base_args))
        args.region = continent

        try:
            (fc, fc_output, obs, lt_idx, doy, train_times, lat_u, lon_u,
             n_lat, n_lon, n_tr_vars, n_out_vars, _) = load_forecasts(
                data_dir, args, lat_vals, lon_vals, train=True,
                patch_num=patch_idx, use_legacy_global_data=USE_LEGACY_GLOBAL_DATA
            )
        except Exception as e:
            print(f"    SKIP (train load error): {e}")
            continue

        try:
            (test_fc, test_fc_out, test_obs, test_lt, test_doy, test_times,
             _, _, _, _, _, _, _) = load_forecasts(
                data_dir, args, lat_vals, lon_vals, train=False,
                patch_num=patch_idx, use_legacy_global_data=USE_LEGACY_GLOBAL_DATA
            )
        except Exception as e:
            print(f"    SKIP (test load error): {e}")
            continue

        # Normalize using training statistics
        stats_in = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
        stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}

        fc_norm = (fc - stats_in['mean']) / stats_in['std']
        fc_out_norm = (fc_output - stats_out['mean']) / stats_out['std']
        obs_norm = (obs - stats_out['mean']) / stats_out['std']

        test_fc_norm = (test_fc - stats_in['mean']) / stats_in['std']
        test_fc_out_norm = (test_fc_out - stats_out['mean']) / stats_out['std']

        region_desc = build_region_descriptor(lat_vals, lon_vals, continent)

        patch_train_data.append({
            'fc_norm': fc_norm, 'fc_out_norm': fc_out_norm, 'obs_norm': obs_norm,
            'lt_idx': lt_idx, 'doy': doy,
            'region_desc': region_desc,
            'stats_in': stats_in, 'stats_out': stats_out,
        })
        patch_test_data.append({
            'test_fc_norm': test_fc_norm, 'test_fc_out_norm': test_fc_out_norm,
            'test_obs': test_obs, 'test_lt': test_lt, 'test_doy': test_doy,
            'test_times': test_times,
            'lat_vals': lat_vals, 'lon_vals': lon_vals,
            'continent': continent, 'patch_idx': patch_idx,
            'stats_out': stats_out,
            'region_desc': region_desc,
            'n_lat': n_lat, 'n_lon': n_lon, 'n_out_vars': n_out_vars,
        })

    n_loaded = len(patch_train_data)
    print(f"\nSuccessfully loaded {n_loaded}/{len(eval_cells)} patches")

    if n_loaded == 0:
        print("ERROR: No patches loaded. Exiting.")
        return

    # ---------------------------------------------------------------------------
    # Phase 2: Build pooled DataLoader and train shared PooledFiLMMLP
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 2: Training pooled PooledFiLMMLP")
    print(f"{'='*60}")

    # Determine dimensions from first patch
    input_dim = patch_train_data[0]['fc_norm'].shape[1]
    output_dim = patch_train_data[0]['fc_out_norm'].shape[1]
    print(f"input_dim={input_dim}, output_dim={output_dim}, n_lead_times={n_lead_times}")

    # Split each patch into train/val (80/20 temporal split)
    all_fc_in_tr, all_fc_out_tr, all_obs_tr = [], [], []
    all_lt_tr, all_doy_tr, all_rdesc_tr = [], [], []
    all_fc_in_val, all_fc_out_val, all_obs_val = [], [], []
    all_lt_val, all_doy_val, all_rdesc_val = [], [], []

    for pd_ in patch_train_data:
        n = pd_['fc_norm'].shape[0]
        split = int(0.8 * n)
        idx = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(idx)
        t_idx, v_idx = idx[:split], idx[split:]

        rdesc_tr = np.tile(pd_['region_desc'][np.newaxis], (len(t_idx), 1))
        rdesc_val = np.tile(pd_['region_desc'][np.newaxis], (len(v_idx), 1))

        all_fc_in_tr.append(pd_['fc_norm'][t_idx])
        all_fc_out_tr.append(pd_['fc_out_norm'][t_idx])
        all_obs_tr.append(pd_['obs_norm'][t_idx])
        all_lt_tr.append(pd_['lt_idx'][t_idx])
        all_doy_tr.append(pd_['doy'][t_idx])
        all_rdesc_tr.append(rdesc_tr)

        all_fc_in_val.append(pd_['fc_norm'][v_idx])
        all_fc_out_val.append(pd_['fc_out_norm'][v_idx])
        all_obs_val.append(pd_['obs_norm'][v_idx])
        all_lt_val.append(pd_['lt_idx'][v_idx])
        all_doy_val.append(pd_['doy'][v_idx])
        all_rdesc_val.append(rdesc_val)

    def make_loader(fc_in_list, fc_out_list, obs_list, lt_list, doy_list, rdesc_list,
                    shuffle=True):
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(np.concatenate(fc_in_list)).float(),
            torch.from_numpy(np.concatenate(fc_out_list)).float(),
            torch.from_numpy(np.concatenate(obs_list)).float(),
            torch.from_numpy(np.concatenate(lt_list)).long(),
            torch.from_numpy(np.concatenate(doy_list)).float(),
            torch.from_numpy(np.concatenate(rdesc_list)).float(),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(
        all_fc_in_tr, all_fc_out_tr, all_obs_tr,
        all_lt_tr, all_doy_tr, all_rdesc_tr, shuffle=True
    )
    val_loader = make_loader(
        all_fc_in_val, all_fc_out_val, all_obs_val,
        all_lt_val, all_doy_val, all_rdesc_val, shuffle=False
    )
    print(f"Pooled train: {len(train_loader.dataset)} samples, "
          f"val: {len(val_loader.dataset)} samples")

    # Build model — smaller hidden_dim than per-patch since data is pooled
    model = PooledFiLMMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=256,
        num_layers=4,
        n_lead_times=n_lead_times,
        lead_time_embedding_dim=lead_time_emb_dim,
        dropout_rate=0.25,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PooledFiLMMLP: {n_params:,} parameters")

    # Train with cosine annealing snapshots (same schedule as block ensemble baseline)
    snapshots, training_time = train_pooled_film_snapshot(
        model, train_loader, val_loader,
        epochs=210, lr=lr, device=device,
        weight_decay=weight_decay, grad_clip=1.0,
        T_0=10, T_mult=1,
    )
    print(f"Training complete: {len(snapshots)} snapshots, {training_time:.2f} min")

    if not snapshots:
        print("ERROR: No snapshots produced (check T_0 vs epochs). Exiting.")
        return

    # ---------------------------------------------------------------------------
    # Phase 3: Evaluate on each patch and write output zarrs
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Phase 3: Per-patch inference and zarr output")
    print(f"{'='*60}")

    for pd_ in patch_test_data:
        continent = pd_['continent']
        patch_idx = pd_['patch_idx']
        lat_vals = pd_['lat_vals']
        lon_vals = pd_['lon_vals']
        region_desc = pd_['region_desc']
        stats_out = pd_['stats_out']
        test_fc_norm = pd_['test_fc_norm']
        test_fc_out_norm = pd_['test_fc_out_norm']
        test_obs = pd_['test_obs']
        test_lt = pd_['test_lt']
        test_doy = pd_['test_doy']
        test_times = pd_['test_times']
        n_lat = pd_['n_lat']
        n_lon = pd_['n_lon']

        # Average predictions across snapshots
        corrections = []
        weights = []
        for snap_state, snap_val_loss in snapshots:
            model.load_state_dict(snap_state)
            corr = apply_film_correction(
                model, test_fc_norm, test_fc_out_norm,
                test_lt, test_doy, region_desc, device
            )
            # Denormalize
            corr = corr * stats_out['std'] + stats_out['mean']
            corrections.append(corr)
            weights.append(1.0 / max(snap_val_loss, 1e-12))

        w = np.array(weights)
        w /= w.sum()
        corrected = np.average(corrections, weights=w, axis=0)

        # Original forecast in physical units
        original = test_fc_out_norm * stats_out['std'] + stats_out['mean']

        # Build output path using same naming convention, with arch='pooled_film'
        args_for_path = SimpleNamespace(
            **vars(base_args),
            region=continent,
            nn_architecture='mlp',
            alternate_loss_fn=None,
            bootstrap=None,
        )
        base_path = os.path.join(output_dir, generate_output_path(args_for_path))
        out_path = base_path.replace('_mlp.zarr', f'_pooled_film_{continent}_bs{patch_idx}.zarr')

        if os.path.exists(out_path):
            print(f"  Skipping {continent} bs{patch_idx} (already exists)")
            continue

        print(f"  Writing {continent} bs{patch_idx} -> {out_path}")
        try:
            save_output(
                output_path=out_path,
                model_name=model_name,
                output_vars=["2m_temperature"],
                lon_values=lon_vals,
                lat_values=lat_vals,
                time_values=test_times,
                lead_times=lead_time_hours,
                original_fc=original,
                corrected_fc=corrected,
                lead_time_indices=test_lt,
                ground_truth_data=test_obs,
                training_time_minutes=training_time,
            )
        except Exception as e:
            print(f"  ERROR saving {continent} bs{patch_idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Pooled FiLM experiment complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
