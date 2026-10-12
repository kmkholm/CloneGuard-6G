"""Package a trained run for the testbed authenticator: weights + meta (shapes, K, validation threshold).
python export_model.py dichasus_dev1 -> emulation/model_<run>.pt"""
import os, sys, json, torch
HERE = os.path.dirname(os.path.abspath(__file__))
run = sys.argv[1]
j = json.load(open(os.path.join(HERE, "..", "results", run + ".json")))
a = j["args"]
st = torch.load(os.path.join(HERE, "..", "runs", run + ".pt"), map_location="cpu")
A = st["tok.ap_emb"].shape[0]
# M, F from the first conv input are not stored; read from the dataset used
from data import load_dichasus, load_rt
D = load_dichasus("cf02") if a["data"] == "dichasus" else load_rt(a["data"])
_, A, M, F = D["H"].shape
thr = j.get("thr_fpr0.1", j["test"][list(j["test"])[0]]["model"]["thr"])     # testbed operates at FPR 0.1 % (validation)
torch.save(dict(state=st, meta=dict(A=A, M=M, F=F, K=a["K"], d=a["d"], pos=a["pos"], forget=bool(a["forget"]),
                                    spatial=a["spatial"], thr=thr, run=run)),
           os.path.join(HERE, "..", "emulation", f"model_{run}.pt"))
print("exported", run, dict(A=A, M=M, F=F, K=a["K"], thr=thr))
