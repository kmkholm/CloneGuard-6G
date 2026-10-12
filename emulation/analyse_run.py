"""Timeline metrics of one testbed run from the collected events (python analyse_run.py out/<run>).
Per clone: t_attach (clone PDU session up), t_first_csi (first clone snapshot scored), t_detect, t_quarantine,
registration outcome of the next clone attempt, data-plane cut (first failed ping after quarantine).
Per legitimate UE: alarms, false quarantines, pings ok, score statistics."""
import os, sys, json, glob, re
import numpy as np

out = sys.argv[1]
ev = []
for f in glob.glob(os.path.join(out, "*", "events.jsonl")):
    for l in open(f):
        try: ev.append(json.loads(l))
        except Exception: pass
ev.sort(key=lambda e: e["t"])
st = json.load(open(os.path.join(out, "state.json")))
supi_idx = {}
for b in st["bindings"]:
    supi_idx.setdefault(b["supi"], []).append((b["gnb"], b["ran_ue"]))
res = dict(run=out, clones=[], legit=[])
clones = [e for e in ev if e["kind"] == "ue_start" and e.get("clone")]
for c in clones:
    supi = "imsi-" + c["imsi"]; idx = c["idx"]; node = c["node"]
    pdu = [e for e in ev if e["kind"] == "ue_pdu" and e["node"] == node and e["idx"] == idx]
    scores = [e for e in ev if e["kind"] == "score" and e.get("supi") == supi and e.get("ran_ue") == idx]
    det = [e for e in ev if e["kind"] == "detect" and e.get("supi") == supi]
    q = [e for e in ev if e["kind"] == "quarantine" and e.get("supi") == supi]
    # credential suspension: the next registration with this SUPI (by the clone or by the victim, both hold it) fails
    rej = [e for e in ev if e["kind"] == "ue_log" and e.get("imsi") == c["imsi"] and "Registration failed" in e["text"]
           and q and e["t"] > q[0]["t"]]
    rej_clone = [e for e in rej if e["node"] == node]
    pings = [e for e in ev if e["kind"] == "ping" and e["node"] == node and e["idx"] == idx]
    first_fail = [p for p in pings if not p["ok"] and q and p["t"] > q[0]["t"]]
    r = dict(supi=supi, node=node, t_start=c["t"], t_pdu=pdu[0]["t"] if pdu else None,
             t_detect=det[0]["t"] if det else None, t_quarantine=q[0]["t"] if q else None,
             detect_gnb=det[0].get("gnb") if det else None, n_scores=len(scores),
             t_reject=rej[0]["t"] if rej else None, t_reject_clone=rej_clone[0]["t"] if rej_clone else None,
             reject_cause=re.search(r"failed \[(\w+)\]", rej[0]["text"]).group(1) if rej else None,
             pings_ok_after_q=sum(p["ok"] for p in pings if q and p["t"] > q[0]["t"]),
             t_first_ping_fail=first_fail[0]["t"] if first_fail else None)
    if r["t_quarantine"] and r["t_first_ping_fail"]:
        r["latency_conn_lost_s"] = r["t_first_ping_fail"] - r["t_quarantine"]
    if r["t_pdu"] and r["t_detect"]:
        r["latency_detect_s"] = r["t_detect"] - r["t_pdu"]
    if r["t_detect"] and r["t_quarantine"]:
        r["latency_quarantine_s"] = r["t_quarantine"] - r["t_detect"]
    res["clones"].append(r)
for u in [e for e in ev if e["kind"] == "ue_start" and not e.get("clone")]:
    supi = "imsi-" + u["imsi"]; idx = u["idx"]; node = u["node"]
    scores = [e for e in ev if e["kind"] == "score" and e.get("supi") == supi and e.get("ran_ue") == idx]
    pings = [e for e in ev if e["kind"] == "ping" and e["node"] == node and e["idx"] == idx]
    det = [e for e in ev if e["kind"] == "detect" and e.get("supi") == supi and e.get("ran_ue") == idx]
    regfail = [e for e in ev if e["kind"] == "ue_log" and e.get("imsi") == u["imsi"] and "Registration failed" in e["text"]]
    res["legit"].append(dict(supi=supi, node=node, n_scores=len(scores), alarms=sum(e["alarm"] for e in scores),
                             reg_failed=len(regfail), victim_of_clone=any(("imsi-" + c["imsi"]) == supi for c in clones),
                             false_detect=len(det), pings=len(pings), ping_ok=sum(p["ok"] for p in pings),
                             logit_mean=float(np.mean([e["logit"] for e in scores])) if scores else None,
                             inf_ms=float(np.mean([e["inf_ms"] for e in scores])) if scores else None))
# health: the largest silence of the authenticator while UEs were attached (a VM stall shows up as a long gap)
sc_t = sorted(e["t"] for e in ev if e["kind"] == "score")
pdu_t = [e["t"] for e in ev if e["kind"] == "ue_pdu"]
res["health"] = dict(n_scores=len(sc_t), max_score_gap_s=float(max(np.diff(sc_t))) if len(sc_t) > 1 else None,
                     t_first_pdu=min(pdu_t) if pdu_t else None, t_last_event=max(e["t"] for e in ev),
                     clones_with_pdu=sum(1 for c in res["clones"] if c["t_pdu"]), n_clones=len(res["clones"]))
res["health"]["ok"] = bool(sc_t) and res["health"]["max_score_gap_s"] < 15 and (res["health"]["n_clones"] == 0 or res["health"]["clones_with_pdu"] >= res["health"]["n_clones"] - 1)
json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
