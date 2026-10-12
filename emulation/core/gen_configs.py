"""Write an all-in-one Open5GS 2.7 configuration (every NF on its own loopback address, NGAP/GTP-U on eth0).
usage (inside the core container): python3 gen_configs.py <core_ip> <mongo_ip> <out_dir>
"""
import os, sys

CORE, DB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
PLMN = "{mcc: '001', mnc: '01'}"
NRF = "client:\n    nrf:\n      - uri: http://127.0.0.10:7777"
LOG = lambda n: f"logger:\n  file:\n    path: /data/{n}.log\n  level: info\nglobal:\n  max:\n    ue: 64\n"


def sbi(addr, extra=""):
    return f"  sbi:\n    server:\n      - address: {addr}\n        port: 7777\n    {NRF}\n{extra}"


cfg = {
    "nrf": LOG("nrf") + f"nrf:\n  serving:\n    - plmn_id: {PLMN}\n  sbi:\n    server:\n      - address: 127.0.0.10\n        port: 7777\n",
    "ausf": LOG("ausf") + "ausf:\n" + sbi("127.0.0.11"),
    "udm": LOG("udm") + "udm:\n" + sbi("127.0.0.12"),
    "udr": f"db_uri: mongodb://{DB}/open5gs\n" + LOG("udr") + "udr:\n" + sbi("127.0.0.20"),
    "pcf": f"db_uri: mongodb://{DB}/open5gs\n" + LOG("pcf") + "pcf:\n" + sbi("127.0.0.13"),
    "bsf": LOG("bsf") + "bsf:\n" + sbi("127.0.0.15"),
    "nssf": LOG("nssf") + "nssf:\n  sbi:\n    server:\n      - address: 127.0.0.14\n        port: 7777\n"
            "    client:\n      nrf:\n        - uri: http://127.0.0.10:7777\n      nsi:\n"
            "        - uri: http://127.0.0.10:7777\n          s_nssai:\n            sst: 1\n",
    "amf": LOG("amf") + "amf:\n" + sbi("127.0.0.5") +
           f"  ngap:\n    server:\n      - address: {CORE}\n"
           f"  guami:\n    - plmn_id: {PLMN}\n      amf_id:\n        region: 2\n        set: 1\n"
           f"  tai:\n    - plmn_id: {PLMN}\n      tac: 1\n"
           f"  plmn_support:\n    - plmn_id: {PLMN}\n      s_nssai:\n        - sst: 1\n"
           "  security:\n    integrity_order: [NIA2, NIA1, NIA0]\n    ciphering_order: [NEA0, NEA1, NEA2]\n"
           "  network_name:\n    full: CloneGuard-6G\n  amf_name: open5gs-amf0\n"
           "  time:\n    t3512:\n      value: 540\n",
    "smf": LOG("smf") + "smf:\n" + sbi("127.0.0.4") +
           "  pfcp:\n    server:\n      - address: 127.0.0.4\n    client:\n      upf:\n        - address: 127.0.0.7\n"
           "  gtpc:\n    server:\n      - address: 127.0.0.4\n  gtpu:\n    server:\n      - address: 127.0.0.4\n"
           "  session:\n    - subnet: 10.45.0.0/16\n      gateway: 10.45.0.1\n  dns:\n    - 8.8.8.8\n  mtu: 1400\n",
    "upf": LOG("upf") + "upf:\n  pfcp:\n    server:\n      - address: 127.0.0.7\n    client:\n"
           f"  gtpu:\n    server:\n      - address: {CORE}\n"
           "  session:\n    - subnet: 10.45.0.0/16\n      gateway: 10.45.0.1\n",
}
for k, v in cfg.items():
    open(os.path.join(OUT, f"{k}.yaml"), "w").write(v)
print("wrote", len(cfg), "configs to", OUT)
