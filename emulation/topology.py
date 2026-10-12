"""Build / start / tear down the 6G clone-attack testbed in GNS3 (REST API, GNS3 VM compute).

Topology (one operator, cell-free cluster of 4 gNB sites sharing one core):
  core ---+--- sw (core L2 segment 10.60.0.0/24) ---+--- gnb1 --- radio1 --- ue1 (legit UEs) + clone1
          |                                          +--- gnb2 --- radio2 --- ue2 + clone2
          +--- pla (authenticator)                   +--- gnb3 ... / gnb4 ...
UERANSIM radio link = UDP over the radio segments; N2 (NGAP/SCTP) and N3 (GTP-U) over the core segment.
Each gNB streams the uplink CSI of its attached UEs (replayed measured channels) to the authenticator.

  python topology.py --build [--sites 4 --ues 2]   # create project + nodes + links, upload data, start
  python topology.py --scenario same_gnb|other_gnb|none --clone-delay 90 --dist 5
  python topology.py --collect out/<run>             # fetch events.jsonl + logs from every node
  python topology.py --stop
"""
import argparse, base64, json, os, secrets, subprocess, sys, time
import requests
import vm_ssh

HERE = os.path.dirname(os.path.abspath(__file__))
INI = os.path.join(os.environ["APPDATA"], "GNS3", "2.2", "gns3_server.ini")
PW = [l.split("=", 1)[1].strip() for l in open(INI) if l.startswith("password")][0]
API = "http://localhost:3080/v2"; AUTH = ("admin", PW)
PROJECT = "SixG-CloneGuard"
MCC, MNC = "999", "70"
STATE = os.path.join(HERE, "project_ids.json")


def api(method, path, **kw):
    r = requests.request(method, API + path, auth=AUTH, timeout=180, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code} {r.text[:300]}")
    return r.json() if r.text else None


def b64(cfg):
    return base64.b64encode(json.dumps(cfg, separators=(",", ":")).encode()).decode()


def docker_node(pid, name, image, cfg, adapters, x, y, symbol):
    return api("POST", f"/projects/{pid}/nodes", json=dict(
        name=name, node_type="docker", compute_id="vm", x=x, y=y, symbol=symbol,
        properties=dict(image=image, adapters=adapters, start_command="python3 /app/node.py", console_type="none",
                        environment="CFG=" + b64(cfg), extra_volumes=["/data"])))["node_id"]


def link(pid, ids, a, ia, b, ib):
    """ia/ib = adapter index for docker nodes, port index for ethernet switches (adapter 0)"""
    def ep(n, i):
        return dict(node_id=ids[n], adapter_number=0, port_number=i) if n.startswith(("sw", "rsw")) else dict(node_id=ids[n], adapter_number=i, port_number=0)
    api("POST", f"/projects/{pid}/links", json=dict(nodes=[ep(a, ia), ep(b, ib)]))


def subscribers(n_sites, n_ues):
    subs = []
    for k in range(1, n_sites + 1):
        for j in range(1, n_ues + 1):
            imsi = f"{MCC}{MNC}{k:02d}{j:08d}"          # 15 digits: MCC(3) MNC(2) MSIN(10)
            subs.append(dict(imsi=imsi, k=secrets.token_hex(16), opc=secrets.token_hex(16), site=k, idx=j,
                             imei=f"35693803{k:02d}{j:05d}"))
    return subs


def build(n_sites, n_ues, scenario, clone_delay, dist_m, seed):
    for p in api("GET", "/projects"):
        if p["name"].startswith(PROJECT):
            for call in (("POST", "/projects/%s/open"), ("POST", "/projects/%s/nodes/stop"), ("POST", "/projects/%s/close"), ("DELETE", "/projects/%s")):
                try: api(call[0], call[1] % p["project_id"])
                except RuntimeError as e: print("note:", str(e)[:100])
    names = {p["name"] for p in api("GET", "/projects")}
    pname = PROJECT if PROJECT not in names else PROJECT + time.strftime("-%m%d-%H%M%S")
    pid = api("POST", "/projects", json=dict(name=pname, auto_close=False, scene_width=2600, scene_height=1200))["project_id"]
    subs = subscribers(n_sites, n_ues)
    core_ip, pla_ip = "10.60.0.1", "10.60.0.100"
    ids = {}
    # switch for the core segment
    ids["sw"] = api("POST", f"/projects/{pid}/nodes", json=dict(name="core-switch", node_type="ethernet_switch", compute_id="vm", x=0, y=-40,
                                                              symbol=":/symbols/affinity/circle/gray/switch.svg"))["node_id"]
    ids["core"] = docker_node(pid, "5gc-open5gs", "sixg-core", dict(role="core", name="core", mcc=MCC, mnc=MNC, pla_ip=pla_ip,
                              ifaces=[dict(dev="eth0", addr=core_ip + "/24")], subscribers=subs), 1, -300, -220,
                              ":/symbols/affinity/circle/green/router_cloud.svg")
    # bindings: (gnb_id, ran_ue index) -> supi, as the UE announces ran_ue = its index within the site
    bindings = []
    import numpy as np
    rng = np.random.default_rng(seed)
    traj_meta = {}
    for k in range(1, n_sites + 1):
        t = np.load(os.path.join(HERE, f"traj_gnb{k}.npz"))
        segs = np.array([u for u in np.unique(t["tid"]) if (t["tid"] == u).sum() >= 600]); rng.shuffle(segs)
        traj_meta[k] = dict(segs=[int(s) for s in segs], pos=t["pos"], tid=t["tid"])
    ids["pla"] = docker_node(pid, "pla-authenticator", "sixg-pla", dict(role="pla", name="pla", core_ip=core_ip, consecutive=3,
                             ifaces=[dict(dev="eth0", addr=pla_ip + "/24")], bindings=bindings), 1, 300, -220,
                             ":/symbols/affinity/circle/red/server.svg")
    link(pid, ids, "sw", 0, "core", 0); link(pid, ids, "sw", 1, "pla", 0)
    for k in range(1, n_sites + 1):          # radio switches first: a clone may attach to another site's switch
        ids[f"rsw{k}"] = api("POST", f"/projects/{pid}/nodes", json=dict(name=f"radio{k}", node_type="ethernet_switch", compute_id="vm", x=-900 + (k - 1) * 600, y=280,
                                                                        symbol=":/symbols/affinity/circle/gray/switch.svg"))["node_id"]
    for k in range(1, n_sites + 1):
        gnb_ip, radio_gnb = f"10.60.0.{10 + k}", f"10.70.{k}.1"
        x = -900 + (k - 1) * 600
        ids[f"gnb{k}"] = docker_node(pid, f"gnb{k}", "sixg-ran", dict(role="gnb", name=f"gnb{k}", mcc=MCC, mnc=MNC, gnb_id=k, amf_ip=core_ip, pla_ip=pla_ip,
                                     srs_period=0.25, ifaces=[dict(dev="eth0", addr=gnb_ip + "/24", netem="delay 2ms 0.5ms"),
                                                              dict(dev="eth1", addr=radio_gnb + "/24")]), 2, x, 120,
                                     ":/symbols/affinity/circle/blue/satellite_dish.svg")
        link(pid, ids, "sw", 1 + k, f"gnb{k}", 0)
        # radio segment: a small switch per site (gNB, UE node, clone node)
        if f"rsw{k}" not in ids:
            ids[f"rsw{k}"] = api("POST", f"/projects/{pid}/nodes", json=dict(name=f"radio{k}", node_type="ethernet_switch", compute_id="vm", x=x, y=280,
                                                                            symbol=":/symbols/affinity/circle/gray/switch.svg"))["node_id"]
        link(pid, ids, f"rsw{k}", 0, f"gnb{k}", 1)
        site_subs = [s for s in subs if s["site"] == k]
        ues = []
        segs = traj_meta[k]["segs"]
        for s in site_subs:
            ues.append(dict(imsi=s["imsi"], k=s["k"], opc=s["opc"], imei=s["imei"], idx=s["idx"], traj=segs[s["idx"] - 1], delay=5 + 3 * s["idx"]))
            bindings.append(dict(gnb=k, ran_ue=s["idx"], supi="imsi-" + s["imsi"]))
        ids[f"ue{k}"] = docker_node(pid, f"ue{k}", "sixg-ran", dict(role="ue", name=f"ue{k}", mcc=MCC, mnc=MNC, gnb_ip=radio_gnb, core_ip=core_ip,
                                    ifaces=[dict(dev="eth0", addr=f"10.70.{k}.21/24")], ues=ues), 1, x - 120, 440,
                                    ":/symbols/affinity/circle/gray/phone_wireless.svg")
        link(pid, ids, f"rsw{k}", 1, f"ue{k}", 0)
        if scenario != "none":
            # clone of UE idx 1 of this site: same SUPI/K/OPc, different IMEI, CSI from another trajectory
            victim = site_subs[0]
            tgt_site = k if scenario == "same_gnb" else (k % n_sites) + 1
            tm = traj_meta[tgt_site]
            vseg = segs[0]
            # choose a start snapshot at ~dist_m from the victim's trajectory start (clone appears elsewhere)
            vpos = traj_meta[k]["pos"][traj_meta[k]["tid"] == vseg][min(clone_delay * 4, 599)] if tgt_site == k else None
            cand = [c for c in tm["segs"] if c != vseg]
            best = None
            for c in cand:
                P = tm["pos"][tm["tid"] == c]
                if vpos is not None:
                    d = np.linalg.norm(P - vpos, axis=1); d = d[:max(1, len(P) - 400)]      # leave >= 400 snapshots for the clone stream
                    i = int(np.argmin(np.abs(d - dist_m)))
                    err = abs(d[i] - dist_m)
                else:
                    i, err = 0, 0.0
                if best is None or err < best[0]:
                    best = (err, c, i)
            cl = dict(imsi=victim["imsi"], k=victim["k"], opc=victim["opc"], imei=f"49015420{k:02d}99999", idx=90 + victim["idx"], traj=best[1],
                      start=best[2], delay=clone_delay, clone=True)
            bindings.append(dict(gnb=tgt_site, ran_ue=90 + victim["idx"], supi="imsi-" + victim["imsi"]))
            cfg = dict(role="ue", name=f"clone{k}", mcc=MCC, mnc=MNC, gnb_ip=f"10.70.{tgt_site}.1", core_ip=core_ip,
                       ifaces=[dict(dev="eth0", addr=f"10.70.{tgt_site}.{60 + k}/24")], ues=[cl])
            ids[f"clone{k}"] = docker_node(pid, f"clone{k}", "sixg-ran", cfg, 1, x + 120, 440, ":/symbols/affinity/circle/red/phone_wireless.svg")
            link(pid, ids, f"rsw{tgt_site}", 2 if tgt_site == k else 3, f"clone{k}", 0)
    # the pla node needs the final binding list -> update its environment
    api("PUT", f"/projects/{pid}/nodes/{ids['pla']}", json=dict(properties=dict(environment="CFG=" + b64(dict(
        role="pla", name="pla", core_ip=core_ip, consecutive=3, ifaces=[dict(dev="eth0", addr=pla_ip + "/24")], bindings=bindings)))))
    api("POST", f"/projects/{pid}/drawings", json=dict(x=-900, y=-330, svg='<svg width="900" height="40"><text font-family="Arial" font-size="22" '
        'font-weight="bold" fill="#1b2631">6G cell-free cluster: Open5GS core + 4 UERANSIM gNB sites, clone-resilient PLA</text></svg>'))
    json.dump(dict(project_id=pid, nodes=ids, scenario=scenario, subs=subs, bindings=bindings), open(STATE, "w"), indent=1)
    print("project", pid, "nodes", len(ids))
    return pid, ids


def upload_data(pid, ids, n_sites):
    """model + trajectory files into the nodes' /data volumes (project-files on the VM)"""
    for name, f in [("pla", "model_final.pt")] + [(f"gnb{k}", f"traj_gnb{k}.npz") for k in range(1, n_sites + 1)]:
        dst = f"/opt/gns3/projects/{pid}/project-files/docker/{ids[name]}/data/"
        remote_tmp = "/home/gns3/sixg_build/" + f
        c = vm_ssh.connect(); s = c.open_sftp(); s.put(os.path.join(HERE, f), remote_tmp); s.close(); c.close()
        tgt = "model.pt" if name == "pla" else "traj.npz"
        print(vm_ssh.run(f"sudo mkdir -p {dst} && sudo cp {remote_tmp} {dst}{tgt} && sudo chmod -R a+rwX {dst} && ls -la {dst}")[1][-200:])


def start(pid, ids):
    api("POST", f"/projects/{pid}/open")
    for name in ["sw"] + [k for k in ids if k.startswith("rsw")] + ["core", "pla"]:
        api("POST", f"/projects/{pid}/nodes/{ids[name]}/start"); print("started", name, flush=True)
    time.sleep(20)
    for name in [k for k in ids if k.startswith("gnb")]:
        api("POST", f"/projects/{pid}/nodes/{ids[name]}/start"); print("started", name, flush=True)
    time.sleep(25)
    for name in [k for k in ids if k.startswith("ue")] + [k for k in ids if k.startswith("clone")]:
        api("POST", f"/projects/{pid}/nodes/{ids[name]}/start"); print("started", name, flush=True)


def collect(out):
    st = json.load(open(STATE)); pid, ids = st["project_id"], st["nodes"]
    os.makedirs(out, exist_ok=True)
    for name, nid in ids.items():
        if name.startswith(("sw", "rsw")): continue
        d = f"/opt/gns3/projects/{pid}/project-files/docker/{nid}/data"
        rc, o, e = vm_ssh.run(f"sudo tar -C {d} -cf /home/gns3/{name}.tar --exclude='*.npz' --exclude='db' --exclude='model.pt' . && sudo chmod a+r /home/gns3/{name}.tar")
        vm_ssh.get(f"/home/gns3/{name}.tar", os.path.join(out, f"{name}.tar"))
        import tarfile; d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        with tarfile.open(os.path.join(out, name + ".tar")) as tf: tf.extractall(d)
    json.dump(st, open(os.path.join(out, "state.json"), "w"), indent=1)
    print("collected", out)


def stop():
    st = json.load(open(STATE)); pid = st["project_id"]
    api("POST", f"/projects/{pid}/nodes/stop"); print("stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true"); ap.add_argument("--sites", type=int, default=4); ap.add_argument("--ues", type=int, default=2)
    ap.add_argument("--scenario", default="same_gnb"); ap.add_argument("--clone-delay", type=int, default=90); ap.add_argument("--dist", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--collect"); ap.add_argument("--stop", action="store_true")
    a = ap.parse_args()
    if a.build:
        pid, ids = build(a.sites, a.ues, a.scenario, a.clone_delay, a.dist, a.seed)
        upload_data(pid, ids, a.sites); start(pid, ids)
    elif a.collect:
        collect(a.collect)
    elif a.stop:
        stop()
