#!/usr/bin/env python3
"""Node agent for the 6G clone-attack testbed (GNS3 docker nodes). Role and parameters come from
the CFG environment variable (base64 JSON).  Roles:
  core : MongoDB + Open5GS 2.7 (NRF SCP AUSF UDM UDR PCF BSF NSSF AMF SMF UPF), subscriber provisioning,
         AMF/SMF event feed to the authenticator (TCP push) and a control port for quarantine commands.
  gnb  : UERANSIM nr-gnb + CSI replay emitter: every attached UE process is bound to a measured trajectory
         and its uplink CSI snapshots are streamed (UDP) to the authenticator, tagged with RAN UE id.
  ue   : UERANSIM nr-ue processes (legitimate UEs or clones) + periodic uplink traffic (ping over PDU).
  pla  : physical-layer authenticator: channel-memory model (PyTorch CPU) per SUPI, decision logic,
         quarantine commands, timeline log.
All events are appended to /data/events.jsonl with the node clock (containers share the VM kernel clock).
"""
import base64, json, os, re, socket, struct, subprocess, sys, threading, time

CFG = json.loads(base64.b64decode(os.environ["CFG"]))
ROLE, NAME = CFG["role"], CFG["name"]
os.makedirs("/data", exist_ok=True)
EV = open("/data/events.jsonl", "a", buffering=1)
LOG = open("/data/node.log", "a", buffering=1)


def log(msg):
    LOG.write(f"{time.time():.3f} {msg}\n"); print(msg, flush=True)


def event(kind, **kw):
    EV.write(json.dumps(dict(t=time.time(), node=NAME, kind=kind, **kw)) + "\n")


def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode:
        log(f"CMD FAILED {cmd}: {r.stderr[:300]}")
    return r.stdout


def setup_net():
    for i in CFG.get("ifaces", []):
        sh(f"ip addr flush dev {i['dev']}; ip addr add {i['addr']} dev {i['dev']}; ip link set {i['dev']} up", True)
        if i.get("netem"):
            sh(f"tc qdisc replace dev {i['dev']} root netem {i['netem']}", True)
    for dst, gw in CFG.get("routes", []):
        sh(f"ip route replace {dst} via {gw}", True)


def tail_follow(path, cb):
    while not os.path.exists(path):
        time.sleep(0.5)
    with open(path, errors="replace") as f:
        while True:
            line = f.readline()
            if line:
                cb(line.rstrip())
            else:
                time.sleep(0.02)


# ----------------------------------------------------------------------------------------- core
OPEN5GS_CFG = {
    "nrf": {"nrf": {"sbi": {"server": [{"address": "127.0.0.10", "port": 7777}]}}},
    "scp": {"scp": {"sbi": {"server": [{"address": "127.0.0.200", "port": 7777}],
                            "client": {"nrf": [{"uri": "http://127.0.0.10:7777"}]}}}},
    "ausf": {"ausf": {"sbi": {"server": [{"address": "127.0.0.11", "port": 7777}],
                              "client": {"scp": [{"uri": "http://127.0.0.200:7777"}]}}}},
    "udm": {"udm": {"sbi": {"server": [{"address": "127.0.0.12", "port": 7777}],
                            "client": {"scp": [{"uri": "http://127.0.0.200:7777"}]}}}},
    "udr": {"udr": {"sbi": {"server": [{"address": "127.0.0.20", "port": 7777}],
                            "client": {"scp": [{"uri": "http://127.0.0.200:7777"}]}}}},
    "pcf": {"pcf": {"sbi": {"server": [{"address": "127.0.0.13", "port": 7777}],
                            "client": {"scp": [{"uri": "http://127.0.0.200:7777"}]}}}},
    "bsf": {"bsf": {"sbi": {"server": [{"address": "127.0.0.15", "port": 7777}],
                            "client": {"scp": [{"uri": "http://127.0.0.200:7777"}]}}}},
    "nssf": {"nssf": {"sbi": {"server": [{"address": "127.0.0.14", "port": 7777}],
                              "client": {"scp": [{"uri": "http://127.0.0.200:7777"}],
                                         "nsi": [{"uri": "http://127.0.0.10:7777", "s_nssai": {"sst": 1}}]}}}},
}


def write_yaml(name, body):
    """minimal YAML writer (dicts, lists, scalars) - enough for Open5GS configs"""
    def dump(o, ind=0):
        p = "  " * ind; out = []
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)) and v:
                    out.append(f"{p}{k}:"); out += dump(v, ind + 1)
                else:
                    out.append(f"{p}{k}: {json.dumps(v) if isinstance(v, str) else v}")
        elif isinstance(o, list):
            for v in o:
                if isinstance(v, dict):
                    lines = dump(v, ind + 1)
                    out.append(f"{p}- " + lines[0].lstrip()); out += lines[1:]
                else:
                    out.append(f"{p}- {v}")
        return out
    body = {"logger": {"file": {"path": f"/opt/open5gs/var/log/open5gs/{name}.log"}}, "global": {"max": {"ue": 1024}}, **body}
    txt = "\n".join(dump(body)) + "\n"
    path = f"/opt/open5gs/etc/open5gs/{name}.yaml"
    open(path, "w").write(txt)
    return path


def core():
    ip = CFG["ifaces"][0]["addr"].split("/")[0]
    plmn = {"mcc": CFG["mcc"], "mnc": CFG["mnc"]}
    sh("sysctl -w net.ipv4.ip_forward=1; ip tuntap add name ogstun mode tun; ip addr add 10.45.0.1/16 dev ogstun; "
       "ip link set ogstun up; iptables -t nat -A POSTROUTING -s 10.45.0.0/16 ! -o ogstun -j MASQUERADE")
    subprocess.Popen("mongod --bind_ip 127.0.0.1 --dbpath /data/db --logpath /data/mongod.log --fork", shell=True).wait()
    for _ in range(60):
        if "ok" in sh("mongo --quiet --eval 'db.runCommand({ping:1})' 2>/dev/null"):
            break
        time.sleep(1)
    # subscribers (open5gs-dbctl document format)
    docs = []
    for s in CFG["subscribers"]:
        docs.append({"imsi": s["imsi"], "subscribed_rau_tau_timer": 12, "network_access_mode": 0,
                     "subscriber_status": 0, "operator_determined_barring": 0, "access_restriction_data": 32,
                     "ambr": {"uplink": {"value": 1, "unit": 3}, "downlink": {"value": 1, "unit": 3}},
                     "security": {"k": s["k"], "amf": "8000", "op": None, "opc": s["opc"], "sqn": 1},
                     "slice": [{"sst": 1, "default_indicator": True, "session": [
                         {"name": "internet", "type": 3, "qos": {"index": 9, "arp": {"priority_level": 8, "pre_emption_capability": 1, "pre_emption_vulnerability": 1}},
                          "ambr": {"uplink": {"value": 1, "unit": 3}, "downlink": {"value": 1, "unit": 3}}}]}],
                     "msisdn": [], "schema_version": 1, "__v": 0})
    # mongo shell parses JSON numbers as doubles; Open5GS expects int32 (NumberInt) and int64 for sqn
    def js(o):
        if isinstance(o, bool): return "true" if o else "false"
        if isinstance(o, int): return f"NumberInt({o})"
        if o is None: return "null"
        if isinstance(o, str): return json.dumps(o)
        if isinstance(o, list): return "[" + ",".join(js(v) for v in o) + "]"
        return "{" + ",".join(f"{json.dumps(k)}:{('NumberLong(%d)' % v) if k == 'sqn' else js(v)}" for k, v in o.items()) + "}"
    open("/data/subs.js", "w").write("db.subscribers.deleteMany({}); db.subscribers.insertMany(" + js(docs) + ");")
    sh("mongo --quiet open5gs /data/subs.js", True)
    log("subscribers: " + sh("mongo --quiet open5gs --eval 'db.subscribers.countDocuments({})'").strip())
    cfgs = dict(OPEN5GS_CFG)
    cfgs["amf"] = {"amf": {"sbi": {"server": [{"address": "127.0.0.5", "port": 7777}], "client": {"scp": [{"uri": "http://127.0.0.200:7777"}]}},
                           "ngap": {"server": [{"address": ip}]},
                           "guami": [{"plmn_id": plmn, "amf_id": {"region": 2, "set": 1}}],
                           "tai": [{"plmn_id": plmn, "tac": 1}],
                           "plmn_support": [{"plmn_id": plmn, "s_nssai": [{"sst": 1}]}],
                           "security": {"integrity_order": ["NIA2", "NIA1", "NIA0"], "ciphering_order": ["NEA0", "NEA1", "NEA2"]},
                           "network_name": {"full": "SixG-CF", "short": "6GCF"}, "amf_name": "amf-cf", "time": {"t3512": {"value": 540}}}}
    cfgs["smf"] = {"smf": {"sbi": {"server": [{"address": "127.0.0.4", "port": 7777}], "client": {"scp": [{"uri": "http://127.0.0.200:7777"}]}},
                           "pfcp": {"server": [{"address": "127.0.0.4"}], "client": {"upf": [{"address": "127.0.0.7"}]}},
                           "gtpc": {"server": [{"address": "127.0.0.4"}]}, "gtpu": {"server": [{"address": "127.0.0.4"}]},
                           "session": [{"subnet": "10.45.0.0/16", "gateway": "10.45.0.1", "dnn": "internet"}],
                           "dns": ["8.8.8.8"], "mtu": 1400}}
    cfgs["upf"] = {"upf": {"pfcp": {"server": [{"address": "127.0.0.7"}]}, "gtpu": {"server": [{"address": ip}]},
                           "session": [{"subnet": "10.45.0.0/16", "gateway": "10.45.0.1", "dnn": "internet", "dev": "ogstun"}]}}
    os.environ["DB_URI"] = "mongodb://127.0.0.1/open5gs"     # the image's env override wins over the yaml
    for nf in ["nrf", "scp", "ausf", "udm", "udr", "pcf", "bsf", "nssf", "amf", "smf", "upf"]:
        body = cfgs[nf]
        if nf in ("udr", "pcf"):
            body = dict(body); body["global"] = {"max": {"ue": 1024}}
        p = write_yaml(nf, body)
        if nf in ("udr", "pcf"):   # db_uri belongs to the top level in 2.7
            open(p, "a").write("db_uri: mongodb://127.0.0.1/open5gs\n")
        subprocess.Popen(f"/opt/open5gs/bin/open5gs-{nf}d -c {p} >> /data/{nf}.out 2>&1", shell=True)
        time.sleep(2 if nf in ("nrf", "scp") else 0.7)
    log("open5gs started on " + ip)
    event("core_up", ip=ip)
    ue_ip = {}

    def amf_line(line):
        # [amf] INFO: [imsi-999700000000001] Registration complete / ... RAN_UE_NGAP_ID[n] ... gNB-N2[ip]
        m = re.search(r"\[(imsi-\d+)\]", line)
        if m and ("Registration complete" in line or "Registration accept" in line or "Deregistration" in line
                  or "Implicit" in line or "Registration reject" in line or "UE Context Release" in line):
            event("amf", supi=m.group(1), text=line[-160:])
        m2 = re.search(r"RAN_UE_NGAP_ID\[(\d+)\].*AMF_UE_NGAP_ID\[(\d+)\]", line)
        if m2:
            event("ngap_ids", ran_ue=int(m2.group(1)), amf_ue=int(m2.group(2)), text=line[-200:])

    def smf_line(line):
        m = re.search(r"UE SUPI\[(imsi-\d+)\] DNN\[\w+\] IPv4\[([\d.]+)\]", line)
        if m:
            ue_ip[m.group(1)] = m.group(2); event("pdu", supi=m.group(1), ip=m.group(2))

    threading.Thread(target=tail_follow, args=("/opt/open5gs/var/log/open5gs/amf.log", amf_line), daemon=True).start()
    threading.Thread(target=tail_follow, args=("/opt/open5gs/var/log/open5gs/smf.log", smf_line), daemon=True).start()

    # event feed: push every event line to the authenticator
    def feed():
        while True:
            try:
                s = socket.create_connection((CFG["pla_ip"], 7002), timeout=10)
                with open("/data/events.jsonl") as f:
                    f.seek(0, 2)
                    while True:
                        l = f.readline()
                        if l:
                            s.sendall(l.encode())
                        else:
                            time.sleep(0.02)
            except Exception as e:
                time.sleep(2)
    threading.Thread(target=feed, daemon=True).start()

    # control port: quarantine / release
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind((ip, 7001)); srv.listen(8)
    while True:
        c, _ = srv.accept()
        try:
            req = json.loads(c.recv(4096).decode())
            supi = req["supi"]; imsi = supi.replace("imsi-", "")
            t0 = time.time()
            if req["cmd"] == "quarantine":
                # (1) data-plane cut of the current PDU session, (2) credential suspension: the subscription is
                # withdrawn from the UDR so that every further registration with this SUPI is rejected
                if supi in ue_ip:
                    sh(f"iptables -I FORWARD -s {ue_ip[supi]} -j DROP; iptables -I FORWARD -d {ue_ip[supi]} -j DROP")
                t1 = time.time()
                sh(f"mongo --quiet open5gs --eval 'var d=db.subscribers.findOne({{imsi:\"{imsi}\"}}); if(d){{db.quarantined.insertOne(d); db.subscribers.deleteOne({{imsi:\"{imsi}\"}})}}'")
                event("quarantine", supi=supi, ip=ue_ip.get(supi), dt_dataplane=t1 - t0, dt=time.time() - t0)
            elif req["cmd"] == "release":
                sh(f"mongo --quiet open5gs --eval 'var d=db.quarantined.findOne({{imsi:\"{imsi}\"}}); if(d){{db.subscribers.insertOne(d); db.quarantined.deleteOne({{imsi:\"{imsi}\"}})}}'")
                if supi in ue_ip:
                    sh(f"iptables -D FORWARD -s {ue_ip[supi]} -j DROP; iptables -D FORWARD -d {ue_ip[supi]} -j DROP")
                event("release", supi=supi)
            c.sendall(b"ok")
        except Exception as e:
            log("ctl error " + str(e))
        c.close()


# ----------------------------------------------------------------------------------------- gnb
def gnb():
    n2 = CFG["ifaces"][0]["addr"].split("/")[0]; radio = CFG["ifaces"][1]["addr"].split("/")[0]
    y = f"""mcc: '{CFG['mcc']}'
mnc: '{CFG['mnc']}'
nci: '0x{CFG['gnb_id']:010x}'
idLength: 32
tac: 1
linkIp: {radio}
ngapIp: {n2}
gtpIp: {n2}
amfConfigs:
  - address: {CFG['amf_ip']}
    port: 38412
slices:
  - sst: 1
ignoreStreamIds: true
"""
    open("/data/gnb.yaml", "w").write(y)
    while "0% packet loss" not in sh(f"ping -c 1 -W 1 {CFG['amf_ip']}"):
        time.sleep(1)
    time.sleep(CFG.get("gnb_delay", 12))

    def run_gnb():
        while True:
            p = subprocess.Popen("nr-gnb -c /data/gnb.yaml >> /data/gnb.out 2>&1", shell=True); p.wait()
            event("gnb_exit", rc=p.returncode); time.sleep(4)
    threading.Thread(target=run_gnb, daemon=True).start()
    event("gnb_up", gnb_id=CFG["gnb_id"])
    ue_ngap = {}       # UERANSIM UE index -> RAN UE NGAP id (from gnb log)

    def gline(line):
        # [2026-..] [rrc] [info] New signal detected for cell[1], total [1] cells in coverage
        # [ngap] [info] UE[1] ... RAN_UE_NGAP_ID or "[rrc] ... UE[1]"
        m = re.search(r"UE\[(\d+)\]", line)
        if m:
            event("gnb_log", ue=int(m.group(1)), text=line[-140:])
        if "SCTP could not connect" in line:          # AMF not up yet: restart nr-gnb (it does not retry NGAP)
            event("gnb_sctp_retry"); sh("pkill -f nr-gnb")
        if "NG Setup procedure is successful" in line:
            event("gnb_ng_setup")
    threading.Thread(target=tail_follow, args=("/data/gnb.out", gline), daemon=True).start()

    # CSI replay emitter: UE processes announce themselves on the radio segment (UDP 4998) with their
    # UE index and trajectory handle; snapshots of that trajectory are streamed to the authenticator.
    import numpy as np
    traj = np.load("/data/traj.npz")           # H (N, A, M, F) complex64 fp16 pairs, tid, pos
    Hr, Hi, tid = traj["Hr"], traj["Hi"], traj["tid"]
    A, M, F = Hr.shape[1:]
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    streams = {}

    def ann():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind((radio, 4998))
        while True:
            d, addr = s.recvfrom(4096); j = json.loads(d.decode())
            key = (addr[0], j["ue"])
            if j.get("stop"):
                streams.pop(key, None); event("stream_stop", ue=j["ue"], src=addr[0]); continue
            if key in streams:            # re-registration of the same UE: the radio channel continues, keep the index
                event("stream_continue", ue=j["ue"], src=addr[0], k=int(streams[key]["k"])); continue
            idx = np.nonzero(tid == j["traj"])[0]
            streams[key] = dict(idx=idx, k=int(j.get("start", 0)), ue=j["ue"], ran_ue=j.get("ran_ue"), t0=time.time())
            event("stream_start", ue=j["ue"], src=addr[0], traj=int(j["traj"]), start=int(j.get("start", 0)), ran_ue=j.get("ran_ue"))
    threading.Thread(target=ann, daemon=True).start()
    period = CFG.get("srs_period", 0.1); seq = 0
    while True:
        t = time.time()
        for key, st in list(streams.items()):
            if st["k"] >= len(st["idx"]):
                continue
            i = st["idx"][st["k"]]; st["k"] += 1; seq += 1
            hdr = struct.pack("<IIIIdH", CFG["gnb_id"], st["ue"], st["ran_ue"] or 0, seq, time.time(), int(i))
            out.sendto(hdr + Hr[i].tobytes() + Hi[i].tobytes(), (CFG["pla_ip"], 7000))
        time.sleep(max(0, period - (time.time() - t)))


# ----------------------------------------------------------------------------------------- ue
def ue():
    radio = CFG["ifaces"][0]["addr"].split("/")[0]
    ann = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for u in CFG["ues"]:
        y = f"""supi: 'imsi-{u['imsi']}'
mcc: '{CFG['mcc']}'
mnc: '{CFG['mnc']}'
key: '{u['k']}'
op: '{u['opc']}'
opType: 'OPC'
amf: '8000'
imei: '{u['imei']}'
imeiSv: '4370816125816151'
gnbSearchList:
  - {CFG['gnb_ip']}
uacAic:
  mps: false
  mcs: false
uacAcc:
  normalClass: 0
  class11: false
  class12: false
  class13: false
  class14: false
  class15: false
sessions:
  - type: 'IPv4'
    apn: 'internet'
    slice:
      sst: 1
configured-nssai:
  - sst: 1
default-nssai:
  - sst: 1
integrity:
  IA1: true
  IA2: true
  IA3: true
ciphering:
  EA1: true
  EA2: true
  EA3: true
integrityMaxRate:
  uplink: 'full'
  downlink: 'full'
"""
        open(f"/data/ue_{u['idx']}.yaml", "w").write(y)

    def run_ue(u):
        time.sleep(u.get("delay", 0))
        event("ue_start", imsi=u["imsi"], idx=u["idx"], clone=u.get("clone", False), traj=u["traj"], start=u.get("start", 0))
        out = f"/data/ue_{u['idx']}.out"; open(out, "a").close()
        p = subprocess.Popen(f"nr-ue -c /data/ue_{u['idx']}.yaml >> {out} 2>&1", shell=True)
        state = dict(pdu=None, traffic=False)

        def ue_line(line):
            low = line.lower()
            if any(k in low for k in ("registration accept", "registration reject", "registration failed", "deregistration", "service reject",
                                      "pdu session establishment is successful", "connection released", "is up.")):
                event("ue_log", imsi=u["imsi"], idx=u["idx"], text=line[-140:])
            m = re.search(r"TUN interface\[(uesimtun\d+), ([\d.]+)\] is up", line)
            if m:
                state["pdu"] = (m.group(1), m.group(2)); event("ue_pdu", imsi=u["imsi"], idx=u["idx"], dev=m.group(1), ip=m.group(2))
                # the UE is now transmitting on the radio: start its CSI stream (gNB-side replay)
                ann.sendto(json.dumps(dict(ue=u["idx"], traj=u["traj"], start=u.get("start", 0), ran_ue=u["idx"])).encode(), (CFG["gnb_ip"], 4998))
                if not state["traffic"]:
                    state["traffic"] = True; threading.Thread(target=traffic, args=(u, state), daemon=True).start()
            if "registration reject" in low or "registration failed" in low:
                ann.sendto(json.dumps(dict(ue=u["idx"], stop=1)).encode(), (CFG["gnb_ip"], 4998))
        threading.Thread(target=tail_follow, args=(out, ue_line), daemon=True).start()
        p.wait()
        event("ue_exit", imsi=u["imsi"], idx=u["idx"], rc=p.returncode)

    def traffic(u, state):
        while True:
            ip = state["pdu"][1] if state["pdu"] else None
            r = sh(f"ping -I {ip} -c 1 -W 1 {CFG['core_ip']}") if ip else ""
            ok = " 0% packet loss" in r
            event("ping", imsi=u["imsi"], idx=u["idx"], ok=ok, src=ip)
            time.sleep(1.0)

    ths = [threading.Thread(target=run_ue, args=(u,), daemon=True) for u in CFG["ues"]]
    for t in ths: t.start()
    while True:
        time.sleep(60)


# ----------------------------------------------------------------------------------------- pla
def pla():
    import numpy as np, torch
    sys.path.insert(0, "/app")
    from model import CloneGuard
    from data import normalise
    ip = CFG["ifaces"][0]["addr"].split("/")[0]
    ck = torch.load("/data/model.pt", map_location="cpu")
    meta = ck["meta"]; K, thr = meta["K"], meta["thr"]
    model = CloneGuard(meta["A"], meta["M"], meta["F"], d=meta["d"], pos=meta["pos"], forget=meta["forget"], spatial=meta["spatial"])
    model.load_state_dict(ck["state"]); model.eval(); torch.set_num_threads(2)
    A, M, F = meta["A"], meta["M"], meta["F"]
    supi_of = {}            # (gnb_id, ran_ue) -> supi, learnt from the core feed
    mem = {}                # supi -> list of accepted snapshot embeddings, max K
    owner = {}              # supi -> RAN stream (supi, gnb, ran_ue) that currently owns the memory
    prob = {}               # stream -> probation logits of a new stream of a known identity
    alarms = {}             # stream -> consecutive alarm count (owner stream, sliding memory)
    quarantined = set()
    m_consec = CFG.get("consecutive", 3); P_prob = CFG.get("probation", 3); prob_min = CFG.get("probation_alarms", 2)
    lock = threading.Lock()

    def feed_srv():
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((ip, 7002)); s.listen(4)
        while True:
            c, _ = s.accept()
            buf = b""
            while True:
                d = c.recv(65536)
                if not d: break
                buf += d
                while b"\n" in buf:
                    l, buf = buf.split(b"\n", 1)
                    try:
                        e = json.loads(l)
                    except Exception:
                        continue
                    if e["kind"] == "ngap_ids":
                        e_last[0] = e
                    if e["kind"] == "amf" and ("Registration complete" in e["text"] or "Registration accept" in e["text"]):
                        # the AMF log prints the RAN UE NGAP id a few lines before; bind the newest pair
                        pass
                    event("core_feed", **{k: v for k, v in e.items() if k not in ("t", "node", "kind")}, k2=e["kind"])
    e_last = [None]
    threading.Thread(target=feed_srv, daemon=True).start()

    # binding RAN stream -> SUPI: the UE announces ran_ue = its UERANSIM index; the core publishes the
    # SUPI with the RAN UE id in its NGAP log (amf). We bind on the core "pdu" event order: simplest
    # robust rule for the testbed - the stream announces carry the SUPI mapping through the config.
    for b in CFG.get("bindings", []):
        supi_of[(b["gnb"], b["ran_ue"])] = b["supi"]

    def ctl(cmd, supi):
        try:
            c = socket.create_connection((CFG["core_ip"], 7001), timeout=5)
            c.sendall(json.dumps(dict(cmd=cmd, supi=supi)).encode()); c.recv(16); c.close()
        except Exception as e:
            log("ctl fail " + str(e))

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20); rx.bind((ip, 7000)); rx.settimeout(0.02)
    hsz = struct.calcsize("<IIIIdH"); n_inf = 0; t_inf = 0.0; n_drop = 0; n_rx = 0
    torch.set_num_threads(CFG.get("threads", 4))
    event("pla_up", K=K, thr=thr)
    while True:
        # drain the socket; keep the newest snapshot per (gnb, RAN UE) so that the authenticator never lags
        pend = {}
        try:
            while True:
                d, _ = rx.recvfrom(65536); n_rx += 1
                gnb_id, ue_idx, ran_ue, seq, ts, i = struct.unpack("<IIIIdH", d[:hsz])
                key = (gnb_id, ran_ue)
                if key in pend: n_drop += 1
                pend[key] = (d, ts, i)
                if len(pend) >= 16: break
        except socket.timeout:
            pass
        items = []
        for (gnb_id, ran_ue), (d, ts, i) in pend.items():
            supi = supi_of.get((gnb_id, ran_ue))
            if supi is None:
                continue
            h = np.frombuffer(d[hsz:hsz + A * M * F * 2], np.float16).astype(np.float32) + 1j * \
                np.frombuffer(d[hsz + A * M * F * 2:], np.float16).astype(np.float32)
            items.append((supi, gnb_id, ran_ue, ts, h.reshape(A, M, F).astype(np.complex64)))
        if not items:
            continue
        t0 = time.time()
        with torch.no_grad():
            # tokenise only the new snapshots; the embeddings of accepted snapshots are cached per identity
            X = np.stack([it[4] for it in items])[:, None]                        # B,1,A,M,F
            Xn, lp = normalise(torch.from_numpy(np.stack([X.real, X.imag], -1)))
            zq = model.snapshot(Xn, lp)[:, 0]                                      # B,d
            batch, Z = [], []
            for k, (supi, gnb_id, ran_ue, ts, h) in enumerate(items):
                hist = mem.setdefault(supi, [])
                if len(hist) < K:
                    hist.append(zq[k]); owner[supi] = (supi, gnb_id, ran_ue); event("enrol", supi=supi, n=len(hist), gnb=gnb_id, ran_ue=ran_ue); continue
                batch.append((supi, gnb_id, ran_ue, ts, zq[k])); Z.append(torch.stack(hist[-K:] + [zq[k]]))
            if not batch:
                continue
            logits = model.forward_from_z(torch.stack(Z))[0].tolist()
        t_inf += time.time() - t0; n_inf += 1
        for (supi, gnb_id, ran_ue, ts, z), logit in zip(batch, logits):
            alarm = logit > thr
            hist = mem[supi]; sk = (supi, gnb_id, ran_ue); declare = False
            if sk == owner.get(supi):
                # established stream: the memory slides; a run of m consecutive alarms declares a hijack
                hist.append(z); del hist[:-K]
                alarms[sk] = alarms.get(sk, 0) + 1 if alarm else 0
                declare = alarms[sk] >= m_consec
            else:
                # a new RAN stream (new registration) of a known identity: probation against the frozen memory
                pl = prob.setdefault(sk, []); pl.append(logit)
                if len(pl) >= P_prob:
                    n_al = sum(l > thr for l in pl[:P_prob])
                    if n_al >= prob_min:
                        declare = True
                    else:                                   # continuity confirmed: the new stream inherits the memory
                        owner[supi] = sk; hist.append(z); del hist[:-K]; alarms[sk] = 0
                        event("handover", supi=supi, gnb=gnb_id, ran_ue=ran_ue, logits=[round(l, 2) for l in pl])
                    prob.pop(sk, None)
            if n_inf % 20 == 0 or sk != owner.get(supi):
                event("score", supi=supi, gnb=gnb_id, ran_ue=ran_ue, logit=logit, alarm=bool(alarm), inf_ms=1000 * t_inf / n_inf,
                      batch=len(batch), drop=n_drop, rx=n_rx, probation=sk != owner.get(supi))
            if declare and supi not in quarantined:
                quarantined.add(supi)
                event("detect", supi=supi, gnb=gnb_id, ran_ue=ran_ue, logit=logit, snapshot_ts=ts)
                ctl("quarantine", supi)
                event("quarantine_sent", supi=supi)


if __name__ == "__main__":
    setup_net()
    event("boot", role=ROLE)
    {"core": core, "gnb": gnb, "ue": ue, "pla": pla}[ROLE]()
