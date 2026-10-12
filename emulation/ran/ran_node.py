"""Radio access node of the GNS3 testbed (UERANSIM).

role gnb : nr-gnb + SRS relay (UE radio link -> MEC, tagged with the gNB UE id) + control agent that executes
           the MEC's containment command (ue-release) through nr-cli.
role ue  : nr-ue with the (possibly cloned) USIM credentials + uplink traffic + SRS sidecar that replays the
           channel this UE would present to the distributed access points (CSI trace replay).
"""
import json, os, re, socket, struct, subprocess, sys, threading, time
import numpy as np
sys.path.insert(0, "/app")
from common import CFG, log, sh, setup_ifaces, spawn, forever

setup_ifaces()
ROLE = CFG["role"]
PLMN = dict(MCC="001", MNC="01")


def write(path, txt):
    open(path, "w").write(txt)


# ------------------------------------------------------------------ gNB
def gnb():
    write("/app/gnb.yaml", f"""mcc: '001'
mnc: '01'
nci: '0x000000010'
idLength: 32
tac: 1
linkIp: {CFG['radio_ip']}
ngapIp: {CFG['bh_ip']}
gtpIp: {CFG['bh_ip']}
amfConfigs:
  - address: {CFG['core_ip']}
    port: 38412
slices:
  - sst: 1
ignoreStreamIds: true
""")
    spawn("nr-gnb", "nr-gnb -c /app/gnb.yaml")
    time.sleep(4)
    name = ""
    for _ in range(30):
        name = sh("nr-cli -d").stdout.strip().split("\n")[0]
        if name: break
        time.sleep(1)
    log("gnb c