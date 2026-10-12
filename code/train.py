"""Train / evaluate the clone-resilient PLA model.

python train.py --data munich28 --tag main_s1 --seed 1
python train.py --data dichasus --pos rope --tag abl_rope_s1
Results -> results/<data>_<tag>.json, predictions -> results/pred/<data>_<tag>.npz, model -> runs/<data>_<tag>.pt
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from data import load_dichasus, load_rt, split_by_traj, subset, EpisodeSampler, normalise, DIST_BINS
from model import CloneGuard

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); os.makedirs(os.path.join(RES, "pred"), exist_ok=True)
RUNS = os.path.join(HERE, "..", "runs"); os.makedirs(RUNS, exist_ok=True)


def get_splits(name):
    if name == "dichasus":
        tr = load_dichasus("cf02"); m_tr, m_va, _ = split_by_traj(tr, (0.85, 0.15, 0.0))
        return dict(train=subset(tr, m_tr), val=subset(tr, m_va), test=load_dichasus("cf03"),
                    test_day2=load_dichasus("cf12")), dict(dist_hi=15.0)
    D = load_rt(name); m = split_by_traj(D)
    return dict(train=subset(D, m[0]), val=subset(D, m[1]), test=subset(D, m[2])), dict(dist_hi=400.0)


def timing_aug(X, g):
    """random UE timing offset: common linear phase across sub-bands, independent per snapshot"""
    B, T, A, M, Fb, _ = X.shape
    tau = (torch.rand(B, T, 1, 1, 1, generator=g, device=X.device) * 2 - 1)
    ph = 2 * np.pi * tau * (torch.arange(Fb, device=X.device) / Fb - 0.5)
    c, s = ph.cos(), ph.sin()
    re, im = X[..., 0], X[..., 1]
    return torch.stack([re * c - im * s, re * s + im * c], -1)


def metrics(y, s, thr=None):
    auc = roc_auc_score(y, s)
    fpr, tpr, th = roc_curve(y, s)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = float((fpr[i] + 1 - tpr[i]) / 2)
    tpr1 = float(np.interp(0.01, fpr, tpr)); tpr01 = float(np.interp(0.001, fpr, tpr))
    if thr is None: thr = float(th[i])
    p = (s >= thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return dict(auc=float(auc), eer=eer, tpr_at_fpr1=tpr1, tpr_at_fpr01=tpr01, thr=thr,
                acc=(tp + tn) / len(y), tpr=tp / max(tp + fn, 1), fpr=fp / max(fp + tn, 1),
                cm=[[tn, fp], [fn, tp]])


@torch.no_grad()
def evaluate(model, S, n, seed, dist_hi, gap=1, snr_db=0.0, bs=512):
    model.eval(); g = torch.Generator(device=S.dev); g.manual_seed(seed)
    ys, ss, ds, sb1, sb2, an, zq = [], [], [], [], [], [], []
    for _ in range(0, n, bs):
        X, y, dist, a = S.sample(bs, g, dist_hi=dist_hi, gap=gap)
        X = S.add_noise(X, g, snr_db)
        Xn, lp = normalise(X)
        logit, _, z = model(Xn, lp)
        # training-free reference tests on the same episodes
        h = torch.complex(Xn[..., 0], Xn[..., 1])
        hq, hl = h[:, -1], h[:, -2]
        num = (hq * hl.conj()).sum((-1, -2)).abs(); den = (hq.abs().pow(2).sum((-1, -2)) * hl.abs().pow(2).sum((-1, -2))).sqrt() + 1e-12
        w = 10 ** (lp[:, -1] / 10)
        sb1.append((1 - (w * num / den).sum(-1)).cpu())                      # power-weighted per-AP correlation test
        sb2.append((lp[:, -1] - lp[:, -2]).abs().mean(-1).cpu())             # AP power-profile (large-scale) test
        ys.append(y.cpu()); ss.append(logit.float().cpu()); ds.append(dist.cpu()); an.append(a.cpu()); zq.append(z.cpu())
    cat = lambda L: torch.cat(L).numpy()
    return dict(y=cat(ys), s=cat(ss), d=cat(ds), corr=cat(sb1), pprof=cat(sb2), a=cat(an), z=cat(zq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dichasus"); ap.add_argument("--tag", default="main_s1")
    ap.add_argument("--K", type=int, default=16); ap.add_argument("--pos", default="path")
    ap.add_argument("--forget", type=int, default=1); ap.add_argument("--spatial", default="attn")
    ap.add_argument("--pred", type=float, default=0.1); ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_test", type=int, default=20000)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    sp, cfg = get_splits(a.data)
    gaps = (1, 2, 4, 8, 16)
    S = {k: EpisodeSampler(v, a.K, dev, gaps=gaps) for k, v in sp.items()}
    A, M, Fb = sp["train"]["H"].shape[1:]
    model = CloneGuard(A, M, Fb, d=a.d, pos=a.pos, forget=bool(a.forget), spatial=a.spatial).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    name = f"{a.data}_{a.tag}"; best, hist = -1, []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        X, y, dist, _ = S["train"].sample(a.bs, g, dist_hi=cfg["dist_hi"])
        if S["train"].noise is not None:
            X = S["train"].add_noise(X, g, float(np.random.uniform(-10, 10)))
        X = timing_aug(X, g)
        Xn, lp = normalise(X)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit, zp, zq = model(Xn, lp)
        logit, zp, zq = logit.float(), zp.float(), zq.float()
        l_cls = F.binary_cross_entropy_with_logits(logit, y.float())
        legit = y == 0
        l_pred = (1 - F.cosine_similarity(zp[legit], zq[legit].detach(), dim=-1)).mean() if legit.any() else 0 * l_cls
        loss = l_cls + a.pred * l_pred
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 50 == 0:
            hist.append(dict(step=step, loss=float(loss), l_cls=float(l_cls), l_pred=float(l_pred)))
        if step % 500 == 0 or step == a.steps:
            ev = evaluate(model, S["val"], 4096, 123, cfg["dist_hi"])
            m = metrics(ev["y"], ev["s"])
            hist[-1].update(val_auc=m["auc"], val_eer=m["eer"])
            print(f"{name} step {step} loss {float(loss):.4f} val auc {m['auc']:.4f} eer {m['eer']:.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if m["auc"] > best:
                best = m["auc"]; torch.save(model.state_dict(), os.path.join(RUNS, name + ".pt"))
    model.load_state_dict(torch.load(os.path.join(RUNS, name + ".pt")))
    ev = evaluate(model, S["val"], 8192, 321, cfg["dist_hi"]); thr = metrics(ev["y"], ev["s"])["thr"]
    # operating thresholds for the testbed: validation legitimate-score quantiles (FPR 1 % and 0.1 %)
    legit_s = np.sort(ev["s"][ev["y"] == 0])
    thr_fpr = {f"thr_fpr{p}": float(legit_s[int((1 - p / 100) * len(legit_s)) - 1]) for p in (1, 0.1)}
    out = dict(args=vars(a), params=npar, train_time_s=time.time() - t0, best_val_auc=best, history=hist, test={},
               val_thr_eer=thr, **thr_fpr)
    for split in [k for k in S if k.startswith("test")]:
        ev = evaluate(model, S[split], a.n_test, 2026, cfg["dist_hi"])
        r = dict(model=metrics(ev["y"], ev["s"], thr), corr_test=metrics(ev["y"], ev["corr"]),
                 power_profile_test=metrics(ev["y"], ev["pprof"]))
        # detection rate per spoofer-distance bin at the validation threshold
        r["by_dist"] = []
        for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
            mm = (ev["y"] == 1) & (ev["d"] >= lo) & (ev["d"] < hi)
            if mm.sum() > 20:
                r["by_dist"].append(dict(lo=lo, hi=hi, n=int(mm.sum()), tpr=float((ev["s"][mm] >= thr).mean())))
        out["test"][split] = r
        np.savez_compressed(os.path.join(RES, "pred", f"{name}_{split}.npz"), **ev)
        print(split, json.dumps({k: round(v, 4) for k, v in r["model"].items() if isinstance(v, float)}),
              "corr auc", round(r["corr_test"]["auc"], 4), "pprof auc", round(r["power_profile_test"]["auc"], 4))
    json.dump(out, open(os.path.join(RES, name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
