"""Testbed campaign: build -> run for `duration` s after the last node start -> collect -> analyse -> stop, for every
configuration. A run whose summary reports health.ok = False (authenticator stalled > 15 s, or a clone that never
obtained a PDU session) is moved to out/failed_<name>_<k> and repeated (up to 3 attempts).
Skips configurations whose out/<name>/summary.json exists.  python run_testbed.py [--dry]"""
import os, sys, json, shutil, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
PY = sys.executable
RUNS = []
for d in (1, 5, 20):
    for s in (1, 2, 3):
        RUNS.append(dict(name=f"same_d{d}_s{s}", sites=4, ues=2, scenario="same_gnb", dist=d, seed=s, clone_delay=60, duration=150))
for s in (1, 2, 3):
    RUNS.append(dict(name=f"other_s{s}", sites=4, ues=2, scenario="other_gnb", dist=5, seed=s, clone_delay=60, duration=150))
for s in (1, 2):
    RUNS.append(dict(name=f"noclone_s{s}", sites=4, ues=2, scenario="none", dist=5, seed=s, clone_delay=60, duration=150))
RUNS.append(dict(name="scale_16ue", sites=4, ues=4, scenario="same_gnb", dist=5, seed=1, clone_delay=60, duration=150))
dry = "--dry" in sys.argv


def one(r):
    out = os.path.join(OUT, r["name"])
    cmd = [PY, "topology.py", "--build", "--sites", str(r["sites"]), "--ues", str(r["ues"]), "--scenario", r["scenario"],
           "--dist", str(r["dist"]), "--seed", str(r["seed"]), "--clone-delay", str(r["clone_delay"])]
    subprocess.run(cmd, cwd=HERE)                       # returns after the last node has been started
    time.sleep(r["duration"] + 15)
    subprocess.run([PY, "topology.py", "--collect", out], cwd=HERE)
    subprocess.run([PY, "analyse_run.py", out], cwd=HERE, stdout=open(os.path.join(out, "analyse.log"), "w"))
    subprocess.run([PY, "topology.py", "--stop"], cwd=HERE)
    json.dump(r, open(os.path.join(out, "config.json"), "w"))
    try:
        h = json.load(open(os.path.join(out, "summary.json")))["health"]
    except Exception:
        h = dict(ok=False)
    print(time.strftime("%H:%M:%S"), r["name"], "health", h, flush=True)
    return h.get("ok", False)


for attempt in range(3):
    for r in RUNS:
        out = os.path.join(OUT, r["name"])
        if os.path.exists(os.path.join(out, "summary.json")):
            continue
        print(time.strftime("%H:%M:%S"), "attempt", attempt, r, flush=True)
        if dry:
            continue
        if not one(r):
            k = 0
            while os.path.exists(os.path.join(OUT, f"failed_{r['name']}_{k}")): k += 1
            shutil.move(out, os.path.join(OUT, f"failed_{r['name']}_{k}")); print("  -> moved to failed, will retry", flush=True)
print("campaign done")
