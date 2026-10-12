"""Experiment grid (fixed up front). Skips runs whose results json exists.
Main: 3 datasets x 3 seeds. Ablations (seed 1): position encoding (rope / none), no forget gate, mean AP pooling,
history length K in {4, 8, 32}, no prediction loss.  python run_queue.py [--dry]"""
import os, sys, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")
DATASETS = ["dichasus", "munich28", "sf7"]
jobs = []
for d in DATASETS:
    for s in (1, 2, 3):
        jobs.append((d, f"main_s{s}", ["--seed", str(s)]))
for d in DATASETS:
    for tag, args in [("abl_rope", ["--pos", "rope"]), ("abl_nopos", ["--pos", "none"]), ("abl_noforget", ["--forget", "0"]),
                      ("abl_meanap", ["--spatial", "mean"]), ("abl_k4", ["--K", "4"]), ("abl_k8", ["--K", "8"]), ("abl_k32", ["--K", "32"]),
                      ("abl_nopred", ["--pred", "0"])]:
        jobs.append((d, tag + "_s1", args + ["--seed", "1"]))
dry = "--dry" in sys.argv
t0 = time.time()
for d, tag, args in jobs:
    out = os.path.join(RES, f"{d}_{tag}.json")
    if os.path.exists(out):
        continue
    if d != "dichasus" and not os.path.exists(os.path.join(HERE, "..", "data", "rt", f"{d}.npz")):
        print("skip (no data yet)", d, tag); continue
    cmd = [sys.executable, "train.py", "--data", d, "--tag", tag] + args
    print(time.strftime("%H:%M:%S"), " ".join(cmd[1:]), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=HERE)
print("queue done", f"{(time.time()-t0)/60:.0f} min")
