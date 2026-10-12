"""Shared helpers for every GNS3 docker node: config from the CFG env var (base64 JSON), interface setup,
logging to /data, background process supervision."""
import base64, json, os, subprocess, threading, time

CFG = json.loads(base64.b64decode(os.environ.get("CFG", "e30=")).decode())
os.makedirs("/data", exist_ok=True)
_logf = open("/data/node.log", "a")


def log(*a):
    s = time.strftime("%H:%M:%S") + " " + CFG.get("name", "?") + " " + " ".join(str(x) for x in a)
    print(s, flush=True); _logf.write(s + "\n"); _logf.flush()


def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        log("CMD FAIL", cmd, r.stderr.strip()[:300])
        if check: raise RuntimeError(cmd)
    return r


def setup_ifaces():
    for itf in CFG.get("ifaces", []):
        for _ in range(120):
            if os.path.exists("/sys/class/net/%s" % itf["dev"]): break
            time.sleep(1)
        d = itf["dev"]
        sh("ip addr flush dev %s" % d)
        sh("ip addr add %s dev %s" % (itf["addr"], d), check=True)
        sh("ip link set %s up" % d)
        sh("ethtool -K %s tso off gso off gro off 2>/dev/null" % d)
        if itf.get("netem"):
            sh("tc qdisc replace dev %s root netem %s" % (d, itf["netem"]))
    for dst, gw in CFG.get("routes", []):
        sh("ip route replace %s via %s" % (dst, gw))


def spawn(name, cmd, restart=True, delay=2, env=None):
    def loop():
        n = 0
        while True:
            with open("/data/%s.log" % name, "a") as lf:
                lf.write("\n=== start %d %s\n" % (n, time.strftime("%F %T"))); lf.flush()
                p = subprocess.Popen(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                     env={**os.environ, **(env or {})})
                rc = p.wait()
                lf.write("=== exit rc=%s %s\n" % (rc, time.strftime("%F %T")))
            n += 1
            if not restart: return
            time.sleep(delay)
    t = threading.Thread(target=loop, daemon=True); t.start(); return t


def forever():
    while True:
        time.sleep(3600)
