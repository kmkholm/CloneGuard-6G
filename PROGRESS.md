# Paper 20 — 6G + AI + cybersecurity (+ GNS3) — PROGRESS (resume file)

Started 2026-10-12 11:10. User request (Arabic): "research on 6G and AI, find simulation data etc. and start";
then "+ cybersecurity", then "+ GNS3 simulation".

## Concept (working; confirm title with user before submission)
Clone-resilient physical-layer authentication (PLA) for cell-free massive-MIMO 6G networks.
Threat: a cloned USIM / stolen credential passes 5G-AKA (cryptography cannot tell twins apart); the clone
transmits from a different place, so the distributed access points see a different multi-AP channel.
AI: a temporal channel-memory model scores each uplink CSI snapshot against the legitimate user's recent
CSI history (mobility-aware, no attacker data needed at training beyond "other positions").
GNS3: Open5GS core + UERANSIM gNB/UEs on the GNS3 VM; clone UE registers with the SAME SUPI/key (AKA
succeeds); a PHY-authenticator service (model) consumes replayed CSI streams and triggers core-side
de-registration -> measure detection/mitigation latency, false-disconnect rate, signalling cost.

## Datasets (>= 2; on D:)
- REAL: DICHASUS cell-free (Univ. Stuttgart, DaRUS) 1.272 GHz, 50 MHz, 4 distributed arrays x 8 ant,
  robot trajectories, tachymeter ground truth: cf02, cf03 (Day 1, doi:10.18419/darus-2854), cf12 (Day 2,
  doi:10.18419/darus-3150, 24 working antennas) -> data/dichasus/*.tfrecords -> code/prep_dichasus.py -> *.npz
- SIM (ray tracing): DeepMIMO v4 (`pip install deepmimo`, run from data/ -> data/deepmimo_scenarios):
  asu_campus_3p5 (downloaded), o1_140 (sub-THz, downloading), planned o1_28 / o1_60 multi-BS (cell-free APs),
  asu_campus_3p5_10cm (fine grid for mobility).
- Windows console: set PYTHONIOENCODING=utf-8 PYTHONUTF8=1 before deepmimo (prints a check-mark).

## Literature gap (Consensus quick scan 2026-10-12)
CF-mMIMO PLA is analytic only (da Silva GLOBECOM 2025 tag-based; Martins WCNC 2025 AP-selection alarms);
learned PLA uses synthetic/stochastic or small Wi-Fi testbeds (Guo TIFS 2025 Siamese; Zeng TWC 2026 CV-AE
clone attacks; Li Access 2021 PC-CVR). None: real measured cell-free CSI + ray-traced 6G bands + core-network
clone-attack emulation. Scopus check still TODO before claiming novelty.

## Environment
- GPU RTX A4000 laptop 8 GB; torch 2.12 cu126; Python 3.13 (C:/Python313).
- GNS3 VM (VirtualBox "GNS3 VM", 12 GB / 8 vCPU, 192.168.56.101, ssh gns3/gns3) - emulation/vm_ssh.py.
  Boot checklist (from DAPPER): set clock; stop/disable apport apt-daily* fwupd-refresh; swapon /opt/swapfile.
- F:/G: drives NOT mounted today -> data kept on D: (61 GB free at start).

## Log
- 11:10 folder created; DeepMIMO v4.0.5 installed; DICHASUS cf02/cf03 downloaded (cf12 downloading).
- 11:35 dichasus_io.py (TF-free tfrecord reader) verified: csi (32,1024,2) f32, pos (3), ~20 Hz.
  prep_dichasus.py running (64 sub-bands). GNS3 VM was powered off -> started headless.

## 2026-10-12 12:50 — session 2 (after restart)
- User: "Scopus is open; prior studies etc.; research topic, no discussion, you write the paper" -> fully autonomous.
- Scopus scan done through the seed tab (lit/scopus_raw.json, lit/scopus_unique.json: 126 unique records, 10 queries).
  Gap confirmed: cell-free PLA = analytic only (da Silva GLOBECOM 2025, Martins WCNC 2025); learned PLA on real
  mobile CSI = Wi-Fi/single AP (Guo TIFS 2025, Li TIFS 2026, Zha TIFS 2025); no core-network clone emulation.
- DICHASUS fixed: Day-1 files interleave two transmit streams (alternate records) + per-record sampling-time
  offset drift -> prep_dichasus.py (STO fix + stream split); lag-1 correlation now 0.85-0.92.
- dichasus_dev1 (K=16, 3000 steps, 33 min): test cf03 AUC 0.998 / EER 0.85 %; day-2 cf12 AUC 0.998 / EER 0.98 %;
  detection 97 % at 25-50 cm spoofer distance; reference tests corr 0.987 / power-profile 0.997 (AUC).
- Sionna RT 2.2 installed (GPU, mitsuba cuda variant); gen_rt.py = Munich 28 GHz + San Francisco 7 GHz, 6 lamp-post
  APs x 8 ant, 32 sub-bands, 240 trajectories x 400 snapshots (ped 1.4 m/s / veh 10 m/s, SRS 20 ms). Pure-python
  Dijkstra was the bottleneck (killed after 1 h) -> traj_fast.py (scipy); full run ~15 min/scene (data/rt/gen.log).
- DeepMIMO daily download cap (~600 MB/day) hit -> DeepMIMO kept only as a cited alternative; Sionna RT is the sim source.
- GNS3: host gns3server.exe started from %APPDATA%\GNS3\2.2 (nohup); VM compute connected (192.168.56.101).
  Images building in VM (/home/gns3/sixg_build/build_*.log): sixg-core (Open5GS 2.7.2 + MongoDB 6 + py3),
  sixg-ran (UERANSIM 3.2.6 + py3/numpy), sixg-pla (python 3.12 + torch CPU). SFTP: use python put() in-process;
  the CLI `vm_ssh.py put` under `timeout` fails with ENOENT.
- emulation/node.py (roles core/gnb/ue/pla), topology.py (--build/--collect/--stop), model_dichasus_dev1.pt,
  traj_gnb1-4.npz (cf12 day-2 segments, never trained on).
- run_queue.py launched 12:50 (results/queue.log): 3 datasets x 3 seeds main + 8 ablations each (steps 2500).

## 2026-10-12 14:00 — testbed works end to end; automated chains running
- Smoke run (1 site, 1 UE + clone, same gNB, 5 m): legit UE scores ~ -5.3 for 83 s (no alarm, 0 drops at 10 Hz);
  clone registers with the same SUPI (AMF accepts, legit UE gets Service Reject) -> PLA detects 0.39 s after the
  clone's PDU session, quarantine 0.09 s later (iptables 9 ms + UDR withdrawal), clone pings fail, the next
  registration with that SUPI is rejected (AMF reject cause 11). Lessons: Debian 11 apt -> archive.debian.org;
  VM CPU has no AVX -> MongoDB 4.4 (buster repo) + legacy `mongo` shell with NumberInt docs; Open5GS reads DB_URI
  env; UERANSIM gNB does not retry NGAP -> pkill/restart on "SCTP could not connect"; IMSI 15 digits, IMEI 15;
  ethernet_switch ports = adapter 0/port k; the PLA must drain the UDP socket and batch (84 ms/inference at 50 Hz
  caused drops + frozen-history runaway false alarms) -> 10 Hz replay + newest-per-UE batching + thr at FPR 0.1 %.
- Background chains (this session): (1) wait sf7.npz -> code/run_queue.py (results/queue.log; gaps now 1-16,
  steps 2500, thr_fpr1/0.1 stored); (2) wait results/dichasus_main_s1.json -> export_model.py -> model_final.pt ->
  rebuild images -> emulation/run_testbed.py (out/<run>/summary.json; out_campaign.log). Campaign = same_gnb
  d{1,5,20} x 3 seeds, other_gnb x 3, noclone x 2 (15 min), scale_16ue.
- Resume: if chains died, rerun `python run_queue.py` (skips done) and `python run_testbed.py` (skips done).
- 14:10 manuscript scaffold: paper/main.tex (elsarticle, Computer Networks), sec_intro/related/method/setup written;
  refs.bib verified on Crossref (code/verify_refs.py; 32 DOIs OK, isecure DOI not on Crossref, 4 no-DOI conf entries).
  figure_src: fig1.py (Figure 1 stage-band, tested with dev1), fig_data.py (Figure 2), fig_results.py (Figures 4-8),
  fig_testbed.py (Figure 9); code/build_tables.py (tab_main/ablation/testbed). Pending: Figure 3 topology + GNS3 screenshot,
  sec_results/discussion/conclusion/abstract after the queue and campaign finish.
- 14:40 USER FEEDBACK: Figure 1 must be an original design (not the stage-band template) -> fig1.py rewritten
  (A measured scene map with snapshot cards, B AP-token encoder with real set-attention graph, C memory ribbon with
  real query attention for legit vs clone, D gauge + core response, E equations, F evaluation). References: all 36
  verified in Scopus (seed tab) + Crossref metadata (paper/refs_check.txt). Memory note: feedback-figure1-original-design.
- 15:05 San Francisco scene: the flat-ground street mask was wrong (hilly scene) -> gen_rt.py street_mask now uses the
  terrain mesh hit (si.shape == ShapePtr(terrain)) + per-cell ground map; AP/UE heights follow the ground. Smoke test OK
  (36k street cells). Full sf7 regeneration launched 15:05 (data/rt/sf7_gen.log; slow under GPU contention with the
  training queue; sf7 queue jobs are skipped while sf7.npz is missing -> rerun run_queue.py afterwards).
- dichasus_main_s1 done (gaps 1-16): cf03 AUC 0.998 / EER 1.18 %; cf12 AUC 0.998 / EER 1.40 %; model_final.pt exported
  (thr at FPR 0.1 %), testbed campaign chain starting.
- 14:52 queue paused (GPU thrash with sf7 generation); waiter restarts run_queue.py when data/rt/sf7.npz appears
- 15:20 first 4-site run: 5 false declarations in 30 s -> cause = authenticator CPU time (422 ms per batch of 8 in the
  VM: the tokenizer re-encoded all K+1 snapshots every time) -> dropped snapshots + frozen memory runaway. Fix: cache the
  snapshot embeddings per identity, tokenise only the query (model.forward_from_z); 32 ms per batch of 8 now.
  collect() now extracts with tarfile (Windows tar --one-top-level silently failed). Campaign restarted 15:20.
- 15:45 ROOT CAUSE of testbed false alarms: cf12 stretches contain sounder record gaps (dt up to 1.2 s) and position
  jumps up to 26 cm (> wavelength) inside the 400-snapshot segments; a frozen memory then alarms forever. data.py
  _segment now cuts at dt > 0.3 s or step > 8 cm (lambda/3); sampler gaps clipped to the trajectory length.
  Testbed: 600-snapshot continuous stretches replayed at 4/s (150 s runs, clone at 60 s). Old DICHASUS results
  moved to results/old_segments; queue retrains dichasus after sf7.npz; chain then exports model_final and runs the
  campaign. USER: methodology figure must be draw.io (paper2drawio-v2 style, little text) -> figure_src/gen_fig1.py.
- 16:20 Figure 1 rebuilt in draw.io (figure_src/gen_fig1.py + figlib.py -> figures/drawio/Figure1.{drawio,pdf,png};
  draw.io desktop export). Manuscript points to figures/drawio/Figure1.pdf. The matplotlib fig1.py is retired.
- 16:45 BUG: the consecutive-alarm counter was keyed by SUPI, so legitimate snapshots of the victim reset the clone's
  alarm run (detections only after the victim stream ended). Now keyed by (SUPI, gNB, RAN UE). Old runs in
  emulation/out_bug_alarmkey. Campaign restarted 16:45 with the retrained model (DICHASUS seeds 1-3: EER 0.7-1.0 %).
  Queue: munich main_s1 started 16:27 (~10 min/run) -> munich/sf7 mains + 24 ablations ~ 5 h.
- 17:05 DECISION POLICY changed (offline-validated on day-2 segments, effective gap 3): the RAN stream that owns the
  identity's memory slides it on every snapshot and a run of m = 3 alarms declares; a NEW RAN stream of a known
  identity (new registration) is tested during a probation of P = 3 snapshots against the frozen memory and declared
  if >= 2 alarm, otherwise it inherits the memory ("handover" event). Offline: 0 false declarations / 2944 legit
  scores (alarm rate 0.4 %), clone probation detection 18/18 (1 m), 45/45 (5 m), 22/22 (20 m); legitimate
  re-registration after a pause passes probation only 31/64 (report honestly). The frozen-memory rule of the first
  campaigns produced runaway false alarms (out_bug_frozen). Health check: stall > 15 s or < n-1 clones with PDU -> retry.
  Method section (sec_method.tex, Algorithm 1) must be updated to this rule before writing results.
- 18:35 New policy works in the testbed: 9 runs (same_gnb d=1/5/20, noclone): detection 0.6-1.3 s after the clone's
  PDU, quarantine 0.12-0.66 s, 0 false declarations in 72 legitimate UE-runs. Fixes: other_gnb scenario crashed
  (radio switch of the target site not yet created -> switches created first); clone start index must leave >= 400
  snapshots (two d=20 clones had 1 snapshot -> runs moved to out_partial and repeated). Figure 1 redrawn in the
  user's reference block-diagram style (dashed layer frames, cubes, row strips, attention grid, zoom-in).
- 18:50 Results so far: DICHASUS AUC 99.9 / EER 0.9 % (all seeds); Munich 28 GHz AUC 92.9 / EER 14.2 %;
  SF 7 GHz AUC 93.1 / EER 13.6 % where the correlation test is better (96.4) -> report honestly; per distance the
  ray-traced scenes are easy >= 5 m (AUC 0.97-0.997) and hard < 1 m (0.6-0.8, vehicles worst) because the specular
  ray-traced channel varies smoothly with position (no diffuse scattering) - code/breakdown.py -> results/breakdown.json.
  fig_results.py / build_tables.py / fig_testbed.py run OK (Figures 4-9, tab_*.tex). Figure 8 t-SNE coloured by logit.
- 19:40 (after context reset) Campaign done: 15 valid runs (same_gnb d=1/5/20 x3, other_gnb x3, noclone x2, scale_16ue);
  failed_* dirs are now EXCLUDED in build_tables.py and fig_testbed.py. analyse_run.py corrected: t_reject = first
  "Initial Registration failed" of the SUPI (clone OR victim) AFTER the quarantine (the old rule counted the
  Service Reject that the duplicate-SUPI registration itself causes, before any quarantine); added
  latency_conn_lost_s (first failed probe after quarantine) and reg_failed/victim_of_clone per legit UE.
  Findings written into sec_results.tex (RESULTSTESTBED filled): 49/49 clones with a user plane declared
  (40 same-gNB incl. 16-UE run, 9 other-gNB); 3 other-gNB clones never got a PDU (SMF silent, T3580 x5, core race);
  detection 0.80+-0.16 s / 0.87+-0.32 s; quarantine 105-958 ms; connectivity lost <= 2.3 s for 37/49, 10-13 s for 12
  (probe goes to the core host -> FORWARD rule not exercised; session removed by the victim's rejected
  re-registration); AMF refused the next registration with the SUPI in 46/49 (28 by the clone itself);
  0/128 legitimate UE-runs declared; inference 18-95 ms (8 UEs), ~90 ms (20 processes). Figure 9 polished.
  sec_setup: 4 snapshots/s (was 10), campaign description added. Abstract/intro/discussion numbers updated.
  REMAINING: queue (munich k8/k32/nopred + sf7 x8, ~20:40) -> breakdown.py, fig_results.py, build_tables.py,
  write RESULTSABL + rewrite discussion "What the memory adds" to the real ablation outcome; final compile.
- 20:12 Figure 1 overlaps fixed (query-flag label, accepted-snapshots box, zoom leaders) and re-exported; Figure 2
  titles/legends fixed; Figure 4/5 legends+titles adjusted in fig_results.py (takes effect at the next run);
  testbed table rebuilt (9 columns, no resizebox); placeins \FloatBarrier before Discussion; memory file updated.
  Waiting for the queue (munich k32 started 20:07 -> nopred -> sf7 x8, ~21:05). Then:
    cd code && python breakdown.py && cd ../figure_src && python fig_results.py && cd ../code && python build_tables.py
    write RESULTSABL in paper/sec_results.tex; rewrite "What the memory adds" in sec_discussion.tex; compile.
- 22:25 QUEUE DONE (368 min). breakdown.py / fig_results.py / build_tables.py rerun with all 35 runs. RESULTSABL
  written honestly: DICHASUS variants all saturate; ray-traced variant spread (~1.5 AUC) ~ seed spread (~1 AUC);
  rotary >= PaTH in AUC, slightly worse at 0.1 % FPR; mean-AP pooling worst in Munich; no-pos never better;
  forget gate / pred loss / K>=16 matter at 0.1 % FPR. Discussion "What the memory adds", intro bullet, abstract
  and conclusion rewritten to this finding (no PaTH superiority claim). Distance paragraph corrected to Figure 6
  (correlation test above CloneGuard at 2-20 m, below it < 1.5 m). Figure 8 title/legend fixed.
  main.pdf = 32 pages, 0 LaTeX errors. MANUSCRIPT COMPLETE pending the user's title/author confirmation.
- 22:40 USER chose the title: "CloneGuard: Clone-Resilient Physical-Layer Authentication for Cell-Free 6G with a
  Learned Channel Memory and a Protocol-Level Core-Network Response" (main.tex updated). Author block still Tawfik/ANU.
