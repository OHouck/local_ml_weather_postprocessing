#!/usr/bin/env python3
"""
finetune.py (post‑processing only)
Author: Ozzy Houck
Split out of the original `finetune.py` on 2025‑04‑27.
Requires that `download_forecasts.py` has already stored the raw *.nc
files under `--data_dir`.
"""

import argparse, os, copy, random, glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import torch, torch.nn as nn, torch.optim as optim

# ───────────────────────────────────────── MODEL ──────────────────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ───────────────────────────────────── ARGUMENTS ─────────────────────────────────────────
REGION_BOUNDS = {
    "india":             (17, 27, 72, 82),
    "usa_south":         (30, 40, -105+360, -95+360),
    "amazon":            (-10, 0, -70+360, -60+360),
    "british_columbia":  (48, 58, -130+360, -120+360),
}
SUBREGION_OFFSETS = {"2x2":4, "4x4":3, "6x6":2, "8x8":1, "10x10":0}

def parse_args():
    p = argparse.ArgumentParser("Fine‑tune regional MLP correction model")
    p.add_argument("--data_dir",  type=str, default="~/wb_finetune_data")
    p.add_argument("--output_dir",type=str, required=True)
    p.add_argument("--model_name", type=str, required=True)  # pangu / ifs / ...
    p.add_argument("--region",     type=str, default="india")
    p.add_argument("--subregion",  type=str, default="10x10")
    p.add_argument("--lead_time_hours", type=int, default=24)
    p.add_argument("--training_vars", nargs="+", default=["2m_temperature"])
    p.add_argument("--output_vars",   nargs="+", default=["2m_temperature"])
    p.add_argument("--train_start", default="2018-01-01")
    p.add_argument("--train_end",   default="2021-12-31")
    p.add_argument("--test_start",  default="2022-01-01")
    p.add_argument("--test_end",    default="2022-12-31")
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--mlp_layers",     type=int, default=5)
    return p.parse_args()

# ─────────────────────────────────── HELPER FUNCTIONS ────────────────────────────────────

def load_combined(root, pattern):
    files = glob.glob(os.path.join(root, "*", pattern))
    if not files:
        raise FileNotFoundError(f"No files in {root} matching {pattern}")
    files.sort()
    return xr.open_mfdataset(files, combine="by_coords", decode_timedelta=True)

def create_loader(x, y, batch):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x).float(),
                                        torch.from_numpy(y).float())
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

def train(model, tr_loader, va_loader, device, lr=1e-5, epochs=1000, patience=50):
    crit = nn.MSELoss(); opt = optim.Adam(model.parameters(), lr=lr)
    best = float("inf"); stall = 0; best_w = copy.deepcopy(model.state_dict())
    for ep in range(1, epochs+1):
        # train
        model.train(); tl=0
        for xb,yb in tr_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tl += loss.item()*xb.size(0)
        tl/=len(tr_loader.dataset)
        # val
        model.eval(); vl=0
        with torch.no_grad():
            for xb,yb in va_loader:
                xb,yb=xb.to(device), yb.to(device)
                vl+=crit(model(xb),yb).item()*xb.size(0)
        vl/=len(va_loader.dataset)
        if vl < best - 1e-12:
            best = vl; stall = 0; best_w = copy.deepcopy(model.state_dict())
        else:
            stall += 1
            if stall >= patience:
                print(f"Early stop @ {ep} (val={vl:.4f})"); break
    model.load_state_dict(best_w); return model

def save_zarr(out_path, name, vars_, lons, lats, times, orig, corr, truth):
    n_v, n_t, n_la, n_lo = len(vars_), len(times), len(lats), len(lons)
    orig = orig.reshape(n_t, n_v, n_la, n_lo).transpose(1,0,2,3)
    corr = corr.reshape(n_t, n_v, n_la, n_lo).transpose(1,0,2,3)
    truth= truth.reshape(n_t,n_v,n_la,n_lo).transpose(1,0,2,3)
    data_vars = {}
    for i,v in enumerate(vars_):
        data_vars[f"{v}_original"]  = (("time","latitude","longitude"), orig[i])
        data_vars[f"{v}_corrected"] = (("time","latitude","longitude"), corr[i])
        data_vars[f"{v}_ground_truth"] = (("time","latitude","longitude"), truth[i])
    ds = xr.Dataset(data_vars, coords={"time":times,"latitude":lats,"longitude":lons})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_zarr(out_path, mode="w")

# ────────────────────────────────────────── MAIN ──────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(58); random.seed(58)
    device = (torch.device('cuda') if torch.cuda.is_available() else
              torch.device('mps')  if torch.backends.mps.is_available() else
              torch.device('cpu'))

    ddir   = os.path.expanduser(args.data_dir)
    odir   = os.path.expanduser(args.output_dir)
    os.makedirs(odir, exist_ok=True)

    lat_min, lat_max, lon_min, lon_max = REGION_BOUNDS[args.region]
    offs = SUBREGION_OFFSETS[args.subregion]
    lat_min, lat_max = lat_min+offs, lat_max-offs+0.25
    lon_min, lon_max = lon_min+offs, lon_max-offs+0.25

    lats = np.arange(lat_min, lat_max, 0.25)
    lons = np.arange(lon_min, lon_max, 0.25)
    tr_times = np.arange(np.datetime64(args.train_start), np.datetime64(args.train_end), np.timedelta64(24,'h'))
    te_times = np.arange(np.datetime64(args.test_start),  np.datetime64(args.test_end),  np.timedelta64(24,'h'))

    # ---- load train ----
    tr_dir = os.path.join(ddir, f"train_{args.region}")
    fc_tr  = load_combined(tr_dir, f"{args.model_name}_train_forecast_data_*.nc")
    obs_tr = load_combined(tr_dir, f"{args.model_name}_train_obs_data_*.nc")

    fc_tr  = fc_tr.sel(time=tr_times, latitude=lats, longitude=lons,
                       prediction_timedelta=np.timedelta64(args.lead_time_hours,'h'))[args.training_vars]
    fc_out = fc_tr.sel(variable=args.output_vars)
    obs_tr = obs_tr.sel(time=tr_times, latitude=lats, longitude=lons)[args.output_vars]

    n_la, n_lo = len(lats), len(lons)
    x = fc_tr.to_array().transpose("time","variable","latitude","longitude").values.reshape(len(tr_times), -1)
    x_out = fc_out.to_array().transpose("time","variable","latitude","longitude").values.reshape(len(tr_times), -1)
    y = obs_tr.to_array().transpose("time","variable","latitude","longitude").values.reshape(len(tr_times), -1)

    mean_x, sd_x = x.mean(0), x.std(0)+1e-8
    mean_y, sd_y = x_out.mean(0), x_out.std(0)+1e-8
    x_n, y_n = (x-mean_x)/sd_x, (y-mean_x)/sd_x

    idx = np.random.permutation(len(tr_times))
    split = int(0.8*len(tr_times))
    tr_loader = create_loader(x_n[idx[:split]], y_n[idx[:split]], batch=32)
    va_loader = create_loader(x_n[idx[split:]], y_n[idx[split:]], batch=32)

    model = SimpleMLP(input_dim=x.shape[1], hidden_dim=args.mlp_hidden_dim,
                      output_dim=y.shape[1], num_hidden_layers=args.mlp_layers).to(device)
    model = train(model, tr_loader, va_loader, device)

    # ---- load test ----
    te_dir = os.path.join(ddir, f"test_{args.region}")
    fc_te  = load_combined(te_dir, f"{args.model_name}_test_forecast_data_*.nc")
    obs_te = load_combined(te_dir, f"{args.model_name}_test_obs_data_*.nc")
    fc_te  = fc_te.sel(time=te_times, latitude=lats, longitude=lons,
                       prediction_timedelta=np.timedelta64(args.lead_time_hours,'h'))[args.training_vars]
    fc_out_te = fc_te.sel(variable=args.output_vars)
    obs_te = obs_te.sel(time=te_times, latitude=lats, longitude=lons)[args.output_vars]

    X_te = fc_te.to_array().transpose("time","variable","latitude","longitude").values.reshape(len(te_times), -1)
    X_out_te = fc_out_te.to_array().transpose("time","variable","latitude","longitude").values.reshape(len(te_times), -1)
    Y_te = obs_te.to_array().transpose("time","variable","latitude","longitude").values.reshape(len(te_times), -1)

    Xn_te = (X_te-mean_x) / sd_x
    with torch.no_grad():
        Yn_hat = model(torch.from_numpy(Xn_te).float().to(device)).cpu().numpy()
    Y_hat = Yn_hat*sd_y + mean_x 

    print("MSE original:", np.mean((X_out_te - Y_te)**2))
    print("MSE corrected:", np.mean((Y_hat   - Y_te)**2))

    # ---- save ----
    tag = f"{args.model_name}/{args.region}/train_{'_'.join(args.training_vars)}_test_{'_'.join(args.output_vars)}_dim{args.subregion}_lead{args.lead_time_hours}h"
    out_path = os.path.join(odir, f"{tag}.zarr")
    save_zarr(out_path, args.model_name, args.output_vars, lons, lats, te_times, X_out_te, Y_hat, Y_te)
    print("✓ Saved", out_path)

if __name__ == "__main__":
    main()