# CloneGuard-6G

Code, data and testbed for **CloneGuard: Clone-Resilient Physical-Layer Authentication for Cell-Free 6G with a
Path-Encoded Forgetting-Attention Channel Memory and a Protocol-Level Core-Network Response** (M. Tawfik, 2026).

A cloned USIM passes 5G-AKA, but the clone transmits from another place: the distributed access points of a
cell-free cluster see a channel that is inconsistent with the recent channel history of the subscription.
CloneGuard scores every uplink channel snapshot of an identity against a learned memory of its accepted snapshots
(set attention over the access points, causal attention with path-encoded keys and a forget gate), and the
declaration is turned into a core-network action (PDU session cut, subscription withdrawn from the UDR) inside a
GNS3 testbed with a real Open5GS core and UERANSIM radio stack.

## Layout

| folder | content |
|---|---|
| `code/` | data preparation (`dichasus_io.py`, `prep_dichasus.py`), Sionna RT generator (`gen_rt.py`, `traj_fast.py`), model (`model.py`), training/evaluation (`train.py`, `run_queue.py`), distance breakdown, tables, reference verification |
| `figure_src/` | all figure scripts (`gen_fig1.py` draws Figure 1 as an editable draw.io file; `fig_*.py` the result figures) |
| `emulation/` | GNS3 testbed: Dockerfiles (`sixg-core`, `sixg-ran`, `sixg-pla`), node agent `node.py`, topology builder `topology.py`, campaign `run_testbed.py`, analyser `analyse_run.py`, trained model `model_final.pt`, replay trajectories `traj_gnb*.npz`, per-run `out/*/summary.json` and `config.json` |
| `results/` | one JSON per training run (35 runs: 3 datasets x 3 seeds + 8 ablations x 3 datasets), `breakdown.json`, `tables.json` |
| `figures/` | rendered figures of the paper |

## Data

* **DICHASUS cell-free measurements** (University of Stuttgart, public): `dichasus-cf02` and `dichasus-cf03`
  at <https://doi.org/10.18419/darus-2854>, `dichasus-cf12` at <https://doi.org/10.18419/darus-3150>.
  Put the `.tfrecords` files in `data/dichasus/` and run `python code/prep_dichasus.py`.
* **Ray-traced scenes** (Munich 28 GHz, San Francisco 7 GHz): generated with Sionna RT 2.2 by
  `python code/gen_rt.py --scene munich28` / `--scene sf7`. The generated files `munich28.npz` and `sf7.npz`
  (about 1.1 GB each) are attached to the GitHub release of this repository; place them in `data/rt/`.

## Reproduce

```bash
pip install torch numpy scipy scikit-learn matplotlib sionna  # Sionna only for gen_rt.py
python code/run_queue.py                 # all training runs -> results/*.json
python code/breakdown.py                 # detection by clone distance
python figure_src/fig_results.py         # Figures 4-8 + results/tables.json
python code/build_tables.py              # LaTeX tables
```

Testbed (GNS3 VM with Docker, `gns3server` reachable at `localhost:3080`):

```bash
cd emulation
docker build -f Dockerfile.core -t sixg-core . ; docker build -f Dockerfile.ran -t sixg-ran . ; docker build -f Dockerfile.pla -t sixg-pla .
python run_testbed.py                    # full campaign -> out/<run>/summary.json
python ../figure_src/fig_testbed.py      # Figure 9
```

## Citation

Please cite the paper when using this code or the generated channel files.
