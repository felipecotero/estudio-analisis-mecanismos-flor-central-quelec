"""Recogida y apertura de un pétalo. Guarda cuadros y curvas para el visor."""
import sys, time, json
import numpy as np
from configs import make
from petal import band, rod

cfg = sys.argv[1]
out = sys.argv[2]
hoop = band(15, 0.8)
TH_OPEN = 12.0
LID = 0.05
FPS = 15
p, m = make(cfg, hoop=hoop, theta_open_deg=TH_OPEN)
thf = float(p.th_open)
PI = np.pi
sm = lambda t: t * t * (3 - 2 * t)
t0 = time.time()

def rec_to(lst):
    def f(state, mean, last):
        lst.append(dict(x=np.asarray(state[0], np.float32), th=float(state[2]), T=float(mean[0]), Qg=float(mean[1]), Ms=float(mean[2]), Mstop=float(mean[3]),
                        vmax=float(np.abs(np.asarray(state[1])).max())))
    return f

# 1. postura abierta
st = p.init_state(thf)
pose = []
st = p.run(st, [thf, 1, 0, 0, 6.0, 10], [thf, 1, 0, 0, 6.0, 10], 5.0, fps=FPS, record=rec_to(pose))
print(cfg, 'abierta: vmax', round(pose[-1]['vmax'], 3), 'Qg', round(pose[-1]['Qg'], 2), 't', round(time.time() - t0), flush=True)
open_pose = pose[-1]
# 2. recoger (cinemático, por tramos con suavizado)
closing = []
T_CLOSE, nseg = 10.0, 40
for k in range(nseg):
    a = thf + (PI - thf) * sm(k / nseg)
    b = thf + (PI - thf) * sm((k + 1) / nseg)
    st = p.run(st, [a, 1, 0, 0, 3.0, 10], [b, 1, 0, 0, 3.0, 10], T_CLOSE / nseg, fps=FPS, record=rec_to(closing))
settle = []
st = p.run(st, [PI, 1, 0, 0, 6.0, 0.6], [PI, 1, 0, 0, 6.0, LID], 3.0, fps=FPS, record=rec_to(settle))   # la capa de encima aplana el vuelo
st = p.run(st, [PI, 1, 0, 0, 8.0, LID], [PI, 1, 0, 0, 8.0, LID], 1.5, fps=FPS, record=rec_to(settle))
stored = settle[-1]
print(cfg, 'recogida: vmax', round(stored['vmax'], 3), 't', round(time.time() - t0), flush=True)
# 3. dimensionar el elástico compensador con el par medido en la recogida
th_c = np.array([c['th'] for c in closing]); Qg_c = np.array([c['Qg'] for c in closing])
sel = th_c > np.deg2rad(97)
need = Qg_c[sel] / (-np.cos(th_c[sel] + p.delta))
M0 = 1.20 * float(need.max())
print(cfg, 'par de gravedad máx al recoger', round(float(Qg_c.max()), 2), 'N·m; al abrir', round(float(Qg_c.min()), 2), '→ M0', round(M0, 2), flush=True)
scale = M0 / p.M0
# 4. apertura dinámica: elástico + rienda que se da a ritmo controlado
opening = []
T_OPEN = 12.0
r = p.sector_r
nseg = 48
for k in range(nseg):
    a = PI - (PI - thf + 0.04) * sm(k / nseg)
    b = PI - (PI - thf + 0.04) * sm((k + 1) / nseg)
    lid_a, lid_b = (LID + 0.25 * k, LID + 0.25 * (k + 1)) if k < 6 else (10, 10)
    st = p.run(st, [0, 0, r * (PI - a), scale, 2.0, lid_a], [0, 0, r * (PI - b), scale, 2.0, lid_b], T_OPEN / nseg, fps=FPS, record=rec_to(opening))
u_end = r * (PI - thf + 0.04)
st = p.run(st, [0, 0, u_end, scale, 3.0, 10], [0, 0, u_end, scale, 3.0, 10], 3.0, fps=FPS, record=rec_to(opening))
print(cfg, 'apertura: th final', round(np.rad2deg(opening[-1]['th']), 2), 'T máx', round(max(o['T'] for o in opening), 1), 't', round(time.time() - t0), flush=True)

def pack(lst):
    return dict(x=np.stack([c['x'] for c in lst]), th=np.array([c['th'] for c in lst]), T=np.array([c['T'] for c in lst]),
                Qg=np.array([c['Qg'] for c in lst]), Ms=np.array([c['Ms'] for c in lst]), Mstop=np.array([c['Mstop'] for c in lst]))

res = dict(open_pose=open_pose['x'], stored=stored['x'], lid=LID, M0=M0, delta=p.delta, sector_r=r, th_open=thf, fps=FPS,
           mass=p.mass_total, clamp=p.clamp_np, m_nodes=p.m_np)
for name, lst in (('closing', closing), ('opening', opening)):
    for k, v in pack(lst).items():
        res[f'{name}_{k}'] = v
# diagnóstico de tensiones del aro
import jax.numpy as jnp
def stress(x, th):
    ki, ko, tw = [np.asarray(a) for a in p.hoop_diag(jnp.asarray(x, jnp.float64), jnp.asarray(th))]
    h = p.hoop
    if h['kind'] == 'fleje':
        b, t = h['b_mm'] / 1e3, h['t_mm'] / 1e3
        sig = h['E'] * (np.abs(ki) * t / 2 + np.abs(ko) * b / 2)
        tau = 80e9 * t * np.abs(tw)
    else:
        d = h['d_mm'] / 1e3
        sig = h['E'] * np.hypot(ki, ko) * d / 2
        tau = 4e9 * d / 2 * np.abs(tw)
    return sig.max() / 1e6, tau.max() / 1e6, np.abs(ki).max(), np.abs(ko).max(), np.abs(tw).max()
smax = max((stress(o['x'], o['th']) for o in opening[::3] + closing[::3]), key=lambda a: a[0])
tmax = max((stress(o['x'], o['th']) for o in opening[::3] + closing[::3]), key=lambda a: a[1])
res['hoop_sigma_max_MPa'], res['hoop_tau_max_MPa'] = smax[0], tmax[1]
print(cfg, 'aro: σ máx', round(smax[0]), 'MPa  τ máx', round(tmax[1]), 'MPa', flush=True)
np.savez_compressed(out, **res)
print(cfg, 'listo en', round(time.time() - t0), 's')
