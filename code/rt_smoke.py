import time
import mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")
import numpy as np
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

print([s for s in dir(rt.scene) if not s.startswith("_")][:60])
scene = load_scene(rt.scene.munich)
scene.frequency = 28e9
mn, mx = scene.mi_scene.bbox().min, scene.mi_scene.bbox().max
print("bbox", mn, mx)
scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=2, num_cols=4, vertical_spacing=0.5, horizontal_spacing=0.5,
                             pattern="tr38901", polarization="V")
# reciprocity: UE as transmitter, APs as receivers (uplink SRS)
for i, p in enumerate([[8.5, 21, 27], [-30, 60, 20], [40, -20, 20], [-60, -40, 25]]):
    scene.add(Receiver(name=f"ap{i}", position=p))
for j in range(64):
    scene.add(Transmitter(name=f"ue{j}", position=[20 + 0.05 * j, 0.0, 1.5]))
solver = PathSolver()
t0 = time.time()
paths = solver(scene, max_depth=3, los=True, specular_reflection=True, diffuse_reflection=False,
               refraction=True, synthetic_array=True, samples_per_src=200000)
freqs = np.linspace(-50e6, 50e6, 64)
h = paths.cfr(frequencies=mi.Float(freqs), normalize_delays=False, out_type="numpy")
print("cfr", h.shape, f"{time.time()-t0:.1f}s", np.abs(h).mean())
