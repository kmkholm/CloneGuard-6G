import sys, time, numpy as np, mitsuba as mi
mi.set_variant("cuda_ad_mono_polarized")
import sionna.rt as rt
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
d=np.load('../data/rt/munich28.npz'); pos=d['pos'][d['tid']==0][:128]; ap=d['ap']
scene=load_scene(rt.scene.munich); scene.frequency=28e9
scene.tx_array=PlanarArray(num_rows=1,num_cols=1,pattern="iso",polarization="V")
scene.rx_array=PlanarArray(num_rows=2,num_cols=4,vertical_spacing=0.5,horizontal_spacing=0.5,pattern="iso",polarization="V")
for i,p in enumerate(ap): scene.add(Receiver(name=f"ap{i}",position=[float(x) for x in p]))
for j,p in enumerate(pos): scene.add(Transmitter(name=f"ue{j}",position=[float(p[0]),float(p[1]),1.5]))
freqs=mi.Float(np.linspace(-50e6,50e6,32))
for sps,diff in [(200000,True),(1000000,True),(1000000,False),(3000000,False)]:
    t0=time.time()
    paths=PathSolver()(scene,max_depth=3,los=True,specular_reflection=True,refraction=False,diffraction=diff,synthetic_array=True,samples_per_src=sps,max_num_paths_per_src=3000,seed=1)
    h=paths.cfr(frequencies=freqs,normalize_delays=False,out_type="numpy"); H=np.transpose(h[:,:,:,0,0,:],(2,0,1,3)).reshape(len(pos),-1)
    H=H/np.linalg.norm(H,axis=1,keepdims=True)
    c=[np.mean([abs(np.vdot(H[k],H[k+l])) for k in range(len(H)-l)]) for l in (1,2,5,20)]
    print(sps,diff,f"{time.time()-t0:.1f}s",np.round(c,3),flush=True)
