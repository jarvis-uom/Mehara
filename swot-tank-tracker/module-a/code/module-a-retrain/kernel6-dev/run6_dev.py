r"""Series 2 development (../PROTOCOL-6.md part 1 section B, ../DEV-6.md), one variant per Kaggle session (5 Nov 2026).

Trains the variant's folds with retrain6.py, one process per fold. With two GPUs (Kaggle's "GPU T4 x2") two folds
train at the same time, one on each GPU; with one GPU they train one after the other. Each fold scores itself on the
development data only. Written to --out:
- fold<k>/: weights/, dev_flood_fold<k-1>.csv, dev_s1s2_fold<k-1>.csv, dw_counts_fold<k-1>.csv, areas_passes/
  (first-half passes only), run_fold<k-1>.json;
- logs/<step>.log and part.json (each step's exit code, minutes and GPU).
A step that fails is logged and the others still run.
    python -u run6_dev.py --variant long --folds 1,2 [--input /kaggle/input] [--out /kaggle/working]
Local check on a CPU (tiny):
    python -B run6_dev.py --variant lovasz --folds 1,2 --dry-run --input <project folder> --out <folder>
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gpu_names():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def install_once():
    """Two folds start at the same moment, and each installs what the image lacks. Installing it here first, once,
    keeps their two pip installs from colliding in the same folder."""
    for module, pkg in (("rasterio", ["rasterio"]),
                        ("segmentation_models_pytorch", ["--no-deps", "segmentation-models-pytorch==0.6.0"])):
        if importlib.util.find_spec(module) is None:
            log(f"installing {pkg[-1]}")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkg], check=True)


def ram_gb():
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            return int(f.readline().split()[1]) / 1e6
    except (OSError, ValueError, IndexError):
        return 0.0


def run_lanes(lanes, out):
    """lanes: {GPU (or None for the CPU): [(step, command), ...]}. Each lane runs its steps one after the other on its
    GPU, and the lanes run at the same time. Each step's output goes to logs/<step>.log and, tagged, to this log."""
    steps = {}

    def lane(gpu, jobs):
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for step, cmd in jobs:
            t0 = time.time()
            with open(out / "logs" / f"{step}.log", "w", encoding="utf-8") as f:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                for line in p.stdout:
                    f.write(line)
                    f.flush()
                    print(f"[{step}] {line}", end="", flush=True)
                p.wait()
            steps[step] = {"exit": p.returncode, "minutes": round((time.time() - t0) / 60, 1), "gpu": gpu}
            log(f"{step}: {'finished' if p.returncode == 0 else 'FAILED, exit ' + str(p.returncode)} after "
                f"{steps[step]['minutes']} min")

    threads = [threading.Thread(target=lane, args=(g, jobs)) for g, jobs in lanes.items() if jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="control, long or lovasz (retrain6.py)")
    ap.add_argument("--folds", default="1,2", help="1 to 5, for example 1,2")
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--dry-run", action="store_true", help="CPU only, a few steps")
    args = ap.parse_args()
    t0 = time.time()
    folds = [int(f) for f in args.folds.split(",")]
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    names = [] if args.dry_run else gpu_names()
    n = min(len(names), 2)
    two = n == 2 and len(folds) > 1 and ram_gb() >= 24
    log(f"series 2 development, variant {args.variant}, folds {folds}: GPUs {names or 'none'}, RAM {ram_gb():.0f} GB; "
        f"{'two folds at a time, one on each GPU' if two else 'one fold at a time'}")
    install_once()
    py, dry = [sys.executable, "-u"], (["--dry-run"] if args.dry_run else [])
    train = [(f"fold{k}", py + [str(HERE / "retrain6.py"), "--variant", args.variant, "--fold", str(k - 1),
                                "--input", args.input, "--out", str(out / f"fold{k}")] + dry) for k in folds]
    steps = run_lanes({0: train[0::2], 1: train[1::2]} if two else {0 if n else None: train}, out)
    need = 1 if args.dry_run else 3
    done = [k for k in folds if len(list((out / f"fold{k}" / "weights").glob(f"fold*_{args.variant}6_seed?.pt"))) == need
            and (out / f"fold{k}" / f"run_fold{k - 1}.json").exists()]
    problems = [s for s, v in steps.items() if v["exit"]] + [f"fold{k}: not finished" for k in folds if k not in done]
    with open(out / "part.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": "PROTOCOL-6.md part 1 B; DEV-6.md", "variant": args.variant, "folds": folds,
                   "gpus": names, "two_at_a_time": two, "ram_gb": round(ram_gb()), "steps": steps, "done": done,
                   "problems": problems, "minutes": round((time.time() - t0) / 60, 1), "dry_run": args.dry_run},
                  f, indent=2)
    log(f"DEVELOPMENT RUN DONE in {(time.time() - t0) / 60:.0f} min: everything finished" if not problems else
        f"DEVELOPMENT RUN ENDED WITH PROBLEMS after {(time.time() - t0) / 60:.0f} min: {problems}")


if __name__ == "__main__":
    main()
