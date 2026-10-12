"""LaTeX tables from results/tables.json and emulation/out/*/summary.json -> paper/tab_*.tex"""
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); P = os.path.join(ROOT, "paper")
T = json.load(open(os.path.join(ROOT, "results", "tables.json")))
DS = [("dichasus", "test", "DICHASUS, day 1 (cf03)"), ("dichasus", "test_day2", "DICHASUS, day 2 (cf12)"), ("munich28", "test", "Munich 28 GHz"), ("sf7", "test", "San Francisco 7 GHz")]


def g(d, tag, split, key, met):
    x = T.get(f"{d}|{tag}|{split}|{key}|{met}")
    return x


def pm(x, scale=100, dec=2):
    if x is None: return "--"
    return f"{scale*x['mean']:.{dec}f}" + (f" $\\pm$ {scale*x['std']:.{dec}f}" if x["n"] > 1 else "")


# Table 2: main results vs reference tests
rows = []
for d, split, name in DS:
    for key, lab in (("model", "CloneGuard"), ("corr_test", "AP-weighted correlation test"), ("power_profile_test", "AP power-profile test")):
        if g(d, "main", split, key, "auc") is None: continue
        rows.append(f"{name if key == 'model' else ''} & {lab} & {pm(g(d,'main',split,key,'auc'))} & {pm(g(d,'main',split,key,'eer'))} & "
                    f"{pm(g(d,'main',split,key,'tpr_at_fpr1'),100,1)} & {pm(g(d,'main',split,key,'tpr_at_fpr01'),100,1)} \\\\")
    rows.append("\\midrule")
open(os.path.join(P, "tab_main.tex"), "w").write(
    "\\begin{table}[!htbp]\\centering\\caption{Clone detection (mean $\\pm$ std over three seeds; AUC, EER and TPR in \\%).}\\label{tab:main}\n"
    "\\small\\begin{tabular}{llcccc}\\toprule Test set & Detector & AUC & EER & TPR@FPR 1\\% & TPR@FPR 0.1\\% \\\\\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\\end{tabular}\\end{table}\n")

# Table 3: ablations
abl = [("main", "CloneGuard (PaTH, forget gate, $K=16$)"), ("abl_rope", "rotary position encoding"), ("abl_nopos", "no position encoding"),
       ("abl_noforget", "no forget gate"), ("abl_meanap", "mean AP pooling (no set attention)"), ("abl_k4", "$K=4$"), ("abl_k8", "$K=8$"), ("abl_k32", "$K=32$"), ("abl_nopred", "no prediction loss")]
rows = []
for tag, lab in abl:
    cells = []
    for d, split, _ in [("dichasus", "test_day2", ""), ("munich28", "test", ""), ("sf7", "test", "")]:
        a, e = g(d, tag, split, "model", "auc"), g(d, tag, split, "model", "eer")
        cells.append(f"{100*a['mean']:.2f} / {100*e['mean']:.2f}" if a else "--")
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
open(os.path.join(P, "tab_ablation.tex"), "w").write(
    "\\begin{table}[!htbp]\\centering\\caption{Ablations: AUC / EER in \\% (CloneGuard: mean of three seeds; variants: seed 1).}\\label{tab:ablation}\n"
    "\\resizebox{\\textwidth}{!}{\\begin{tabular}{lccc}\\toprule Variant & DICHASUS day 2 & Munich 28 GHz & SF 7 GHz \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")

# Table 4: testbed
runs = []
for f in sorted(glob.glob(os.path.join(ROOT, "emulation", "out", "*", "summary.json"))):
    if os.path.basename(os.path.dirname(f)).startswith("failed_"): continue
    s = json.load(open(f)); c = json.load(open(os.path.join(os.path.dirname(f), "config.json")))
    runs.append((c, s))
rows = []; agg = {}
for c, s in runs:
    key = (c["scenario"], c["dist"], c["ues"])
    a = agg.setdefault(key, dict(det=[], q=[], cl=[], rej=0, n_clone=0, n_pdu=0, false=0, legit=0, inf=[]))
    for cl in s["clones"]:
        a["n_clone"] += 1
        if cl.get("t_pdu") is None: continue
        a["n_pdu"] += 1
        if cl.get("latency_detect_s") is not None: a["det"].append(cl["latency_detect_s"])
        if cl.get("latency_quarantine_s") is not None: a["q"].append(cl["latency_quarantine_s"])
        if cl.get("latency_conn_lost_s") is not None: a["cl"].append(cl["latency_conn_lost_s"])
        if cl.get("t_reject"): a["rej"] += 1
    for l in s["legit"]:
        a["legit"] += 1; a["false"] += l["false_detect"]
        if l.get("inf_ms"): a["inf"].append(l["inf_ms"])
order = {"none": 0, "same_gnb": 1, "other_gnb": 2}
EOL = "\\\\"
for (sc, dist, ues), a in sorted(agg.items(), key=lambda kv: (order[kv[0][0]], kv[0][2], kv[0][1])):
    name = {"same_gnb": "victim's gNB", "other_gnb": "another gNB", "none": "no clone"}[sc]
    if sc == "none":
        rows.append(f"{name} & -- & {ues*4} & -- & -- & -- & -- & -- & {a['false']}/{a['legit']} " + EOL); continue
    det = f"{np.mean(a['det']):.2f} $\\pm$ {np.std(a['det']):.2f}"
    q = f"{1000*np.mean(a['q']):.0f} ({1000*min(a['q']):.0f}--{1000*max(a['q']):.0f})"
    cl = f"{np.median(a['cl']):.1f} ({min(a['cl']):.1f}--{max(a['cl']):.1f})"
    rows.append(f"{name} & {dist} & {ues*4} & {len(a['det'])}/{a['n_pdu']} ({a['n_clone']}) & {det} & {q} & {cl} & {a['rej']}/{a['n_pdu']} & {a['false']}/{a['legit']} " + EOL)
CAP = ("Testbed campaign: declared / clones with a PDU session (clones started); detection latency from the clone's PDU session; quarantine latency from the declaration; connectivity loss from the quarantine (median, range); SUPIs whose next registration was refused; false declarations over legitimate UE-runs.")
open(os.path.join(P, "tab_testbed.tex"), "w").write(
    "\\begin{table}[!htbp]\\centering\\caption{" + CAP + "}\\label{tab:testbed}\n"
    "\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\\begin{tabular}{lcccccccc}\\toprule Clone at & $d$ (m) & UEs & declared & detection (s) & quarantine (ms) & conn.\\ lost (s) & re-reg.\\ refused & false decl. " + EOL + "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
print("tables written:", len(runs), "testbed runs")
