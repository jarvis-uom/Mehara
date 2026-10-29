"""Member C pilot, part 2: the LSTM named in the plan, on the same rows as the gradient-boosting test.
Input: the last 60 days of storage, its daily change, the seasonal average, rain and the season; plus capacity,
type and the seasonal-average change to each target day. Output: how far the storage change at 7, 14 and 28 days
differs from the seasonal-average change. Dropout, and early stopping on the last training year."""
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view

from c_prep import CUT, HORIZONS, features, panel

sys.stdout.reconfigure(encoding="utf-8")
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
L = 60

p = panel()
seqs, static = {}, {}
for (kind, name), g in p.groupby(["kind", "reservoir"], sort=False):
    g = g.set_index("date")
    doy = g.index.dayofyear.values
    filled = g.frac.ffill().fillna(0)
    arr = np.stack([filled.values, (filled.diff().fillna(0).clip(-0.2, 0.2) * 10).values, g.frac.notna().values.astype(float),
                    g.lta.fillna(g.lta.mean()).values, np.log1p(g.rain.fillna(0).values) / 3,
                    np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], axis=1).astype(np.float32)
    arr = np.vstack([np.zeros((L - 1, arr.shape[1]), np.float32), arr])
    seqs[(kind, name)] = (pd.Series(np.arange(len(g)), index=g.index), sliding_window_view(arr, (L, arr.shape[1]))[:, 0])
    static[(kind, name)] = [np.log10(g.cap.iloc[0]) - 4, float(kind == "Major")]

tabs = [features(p, h)[["kind", "reservoir", "date", "y", "dlta"]].rename(columns={"y": f"y{h}", "dlta": f"dlta{h}"})
        .set_index(["kind", "reservoir", "date"]) for h in HORIZONS]
u = pd.concat(tabs, axis=1).reset_index()
u["test"] = u.date >= CUT


def tensors(rows):
    X = np.empty((len(rows), L, 7), np.float32)
    S = np.empty((len(rows), 2 + len(HORIZONS)), np.float32)
    for (kind, name), g in rows.groupby(["kind", "reservoir"], sort=False):
        pos, win = seqs[(kind, name)]
        ii = np.flatnonzero((rows.kind.values == kind) & (rows.reservoir.values == name))
        X[ii] = win[pos.loc[g.date].values]
        S[ii, :2] = static[(kind, name)]
    S[:, 2:] = rows[[f"dlta{h}" for h in HORIZONS]].fillna(0).values
    Y = (rows[[f"y{h}" for h in HORIZONS]].values - S[:, 2:]).astype(np.float32)   # residual over the seasonal change
    return torch.tensor(X), torch.tensor(S), torch.tensor(np.nan_to_num(Y)), torch.tensor(~np.isnan(Y))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(7, 48, batch_first=True)
        self.drop = torch.nn.Dropout(0.2)
        self.head = torch.nn.Sequential(torch.nn.Linear(48 + 2 + len(HORIZONS), 64), torch.nn.ReLU(),
                                        torch.nn.Dropout(0.2), torch.nn.Linear(64, len(HORIZONS)))

    def forward(self, x, s):
        _, (hid, _) = self.lstm(x)
        return self.head(torch.cat([self.drop(hid[-1]), s], 1))


tr, te = u[~u.test].reset_index(drop=True), u[u.test].reset_index(drop=True)
va_mask = (tr.date >= CUT - pd.Timedelta(days=365)).values
Xtr, Str, Ytr, Mtr = tensors(tr)
Xte, Ste, Yte, Mte = tensors(te)
iv, it = np.flatnonzero(va_mask), np.flatnonzero(~va_mask)
print(f"train sequences {len(it):,} | validation (last training year) {len(iv):,} | test sequences {len(te):,}")


def predict(X, S):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(X[i:i + 4096], S[i:i + 4096]) for i in range(0, len(X), 4096)])


net = Net()
opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
best, best_state, wait = 1e9, None, 0
for epoch in range(25):
    t0, perm, tot = time.time(), torch.tensor(it)[torch.randperm(len(it))], 0.0
    net.train()
    for i in range(0, len(perm), 1024):
        b = perm[i:i + 1024]
        loss = ((net(Xtr[b], Str[b]) - Ytr[b]).abs() * Mtr[b]).sum() / Mtr[b].sum()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    pv = predict(Xtr[iv], Str[iv])
    val = (((pv - Ytr[iv]).abs() * Mtr[iv]).sum() / Mtr[iv].sum()).item()
    print(f"epoch {epoch + 1}: train {tot / len(it) * 100:.2f} | validation {val * 100:.2f} (% of capacity, {time.time() - t0:.0f} s)")
    if val < best - 1e-5:
        best, best_state, wait = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= 3:
            break
net.load_state_dict(best_state)
pred = (predict(Xte, Ste) + Ste[:, 2:]).numpy()   # add the seasonal change back

gb = pd.read_csv("c_test_predictions.csv", parse_dates=["date"])
rows = []
for j, h in enumerate(HORIZONS):
    m = te[[f"y{h}"]].notna().values[:, 0]
    lstm = te.loc[m, ["kind", "reservoir", "date"]].assign(lstm=pred[m, j])
    t = gb[gb.h_days == h].merge(lstm, on=["kind", "reservoir", "date"])
    row = {"h_days": h, "rows": len(t)}
    for c in ["persist", "seasonal_change", "gbm", "lstm"]:
        row[c] = round(((t[c] - t.y).abs() * 100).mean(), 2)
    row["lstm_unseen"] = None
    rows.append(row)
r = pd.DataFrame(rows)
r.to_csv("c_lstm_results.csv", index=False)
print("\nMAE in % of capacity, same test rows (1 Apr 2025 - 27 Oct 2026):")
print(r.drop(columns="lstm_unseen").to_string(index=False))
