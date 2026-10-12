"""Subscriber database node (MongoDB) provisioned with the 5G subscriptions used in the experiment.
The legitimate UE and its clone share ONE subscription (same SUPI, K and OPc): a cloned USIM."""
import sys, time
sys.path.insert(0, "/app")
from common import CFG, log, sh, setup_ifaces, spawn, forever

setup_ifaces()
spawn("mongod", "mongod --bind_ip_all --dbpath /data/db", restart=True)
sh("mkdir -p /data/db")
for _ in range(60):
    if sh("mongosh --quiet --eval 'db.runCommand({ping:1}).ok'").stdout.strip() == "1": break
    time.sleep(2)


def sub_doc(imsi, k, opc):
    return """{imsi: '%s', msisdn: [], imeisv: [], mme_host: [], mme_realm: [], purge_flag: [],
 security: {k: '%s', op: null, opc: '%s', amf: '8000', sqn: NumberLong(1)},
 ambr: {downlink: {value: 1, unit: 3}, uplink: {value: 1, unit: 3}},
 slice: [{sst: 1, default_indicator: true, session: [{name: 'internet', type: 3,
   qos: {index: 9, arp: {priority_level: 8, pre_emption_capability: 1, pre_emption_vulnerability: 1}},
   ambr: {downlink: {value: 1, unit: 3}, uplink: {value: 1, unit: 3}}, pcc_rule: []}]}],
 access_restriction_data: 32, subscriber_status: 0, network_access_mode: 0,
 subscribed_rau_tau_timer: 12, __v: 0}""" % (imsi, k, opc)


for s in CFG["subscribers"]:
    js = "db.subscribers.deleteMany({imsi:'%s'}); db.subscribers.insertOne(%s)" % (s["imsi"], sub_doc(s["imsi"], s["k"], s["opc"]))
    r = sh("mongosh open5gs --quiet --eval \"%s\"" % js.replace("\n", " "))
    log("provisioned", s["imsi"], r.stdout.strip()[:120])
forever()
