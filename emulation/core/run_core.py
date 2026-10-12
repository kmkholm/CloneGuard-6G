"""5G standalone core node: all Open5GS network functions in one container (loopback SBI, NGAP/GTP-U on eth0)."""
import sys, time
sys.path.insert(0, "/app")
from common import CFG, log, sh, setup_ifaces, spawn, forever

setup_ifaces()
core_ip, db_ip = CFG["core_ip"], CFG["db_ip"]
sh("python3 /app/gen_configs.py %s %s /app/cfg" % (core_ip, db_ip), check=True)
for i in (4, 5, 7, 10, 11, 12, 13, 14, 15, 20):
    sh("ip addr add 127.0.0.%d/8 dev lo 2>/dev/null" % i)
# user-plane tunnel for the PDU sessions
sh("ip tuntap add name ogstun mode tun 2>/dev/null")
sh("ip addr add 10.45.0.1/16 dev ogstun 2>/dev/null"); sh("ip link set ogstun up")
sh("sysctl -w net.ipv4.ip_forward=1")
for _ in range(60):                                   # wait for MongoDB
    if sh("timeout 2 bash -c '</dev/tcp/%s/27017'" % db_ip).returncode == 0: break
    time.sleep(2)
B = "/opt/open5gs/bin/open5gs-%sd -c /app/cfg/%s.yaml"
spawn("nrf", B % ("nrf", "nrf")); time.sleep(3)
for nf in ("ausf", "udm", "udr", "pcf", "bsf", "nssf", "upf", "smf"):
    spawn(nf, B % (nf, nf)); time.sleep(1)
time.sleep(3)
spawn("amf", B % ("amf", "amf"))
# capture control-plane signalling (NGAP over SCTP) on the backhaul for latency measurements
spawn("pcap_n2", "tcpdump -i eth0 -n -s 256 -w /data/n2.pcap sctp", restart=False)
log("core up", core_ip)
forever()
