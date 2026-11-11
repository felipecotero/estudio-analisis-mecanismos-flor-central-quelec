"""Opción B: aro + cruceta + flex en todo el contorno. Contención con cordón, volteo y desplegado."""
import sys, time
import numpy as np
import jax.numpy as jnp
from configs import make, make_B, boundary_loop, nodes_on
from petal import band, rod, GFRP

out = sys.argv[1]; d_per = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TH_OPEN, FPS, LID = 12.0, 15, 2.0
p, m = make_B(theta_open_deg=TH_OPEN)
AX = nodes_on(m['P'], y=0.0, lo=1.875, hi=4.0)
P = m['P']; loop = boundary_loop(m)
thf = float(p.th_open); PI = np.pi
sm = lambda t: t * t * (3 - 2 * t)
t0 = time.time()

def rec_to(lst):
    def f(state, mean, last):
        lst.append(dict(x=np.asarray(state[0], np.float32), th=float(state[2]), T=float(mean[0]), Qg=float(mean[1]), Ms=float(mean[2]), Tc=float(mean[4]),
                        vmax=float(np.abs(np.asarray(state[1])).max())))
    return f

def rod_strain(x):
    r = x[np.r_[loop, loop[0], loop[1]]]; e = np.diff(r, axis=0); le = np.linalg.norm(e, axis=1); tt = e / le[:, None]
    kap = np.linalg.norm(np.diff(tt, axis=0), axis=1) / (0.5 * (le[:-1] + le[1:]))
    ra = x[AX]; ea = np.diff(ra, axis=0); la = np.linalg.norm(ea, axis=1); ta = ea / la[:, None]
    ka = np.linalg.norm(np.diff(ta, axis=0), axis=1) / (0.5 * (la[:-1] + la[1:]))
    return float(max(kap.max(), ka.max()) * d_per / 2e3)

def seg(st, lst, n, T, f0, f1, extra):
    """Interpola theta y cordón con suavizado; extra = (cinemático, cuerda(theta) o None, resorte, amortiguación, tapa0, tapa1)."""
    for k in range(n):
        a, b = sm(k / n), sm((k + 1) / n)
        c0, c1 = f0(a), f1(b)
        st = p.run(st, c0, c1, T / n, fps=FPS, record=rec_to(lst))
    return st

st = p.init_state(thf)
pose = []
st = p.run(st, [thf, 1, 0, 0, 6.0, 10, 1], [thf, 1, 0, 0, 6.0, 10, 1], 6.0, fps=FPS, record=rec_to(pose))
open_pose = pose[-1]
print('abierta vmax', round(open_pose['vmax'], 3), 'Qg', round(open_pose['Qg'], 1), 't', round(time.time() - t0), flush=True)
# contener: el cordón recoge el contorno bajo el disco
cinching = []
n = 30
for k in range(n):
    a, b = 1 - sm(k / n), 1 - sm((k + 1) / n)
    st = p.run(st, [thf, 1, 0, 0, 4.0, 10, a], [thf, 1, 0, 0, 4.0, 10, b], 7.0 / n, fps=FPS, record=rec_to(cinching))
st = p.run(st, [thf, 1, 0, 0, 6.0, 10, 0], [thf, 1, 0, 0, 6.0, 10, 0], 2.0, fps=FPS, record=rec_to(cinching))
print('contenida vmax', round(cinching[-1]['vmax'], 3), 'Tc máx', round(max(c['Tc'] for c in cinching), 1), 'Tc fin', round(cinching[-1]['Tc'], 1),
      'deformación varilla máx', round(max(rod_strain(c['x']) for c in cinching[::5]), 4), 't', round(time.time() - t0), flush=True)
# voltear a la pila
closing = []
n = 40
for k in range(n):
    a = thf + (PI - thf) * sm(k / n); b = thf + (PI - thf) * sm((k + 1) / n)
    st = p.run(st, [a, 1, 0, 0, 3.0, 10, 0], [b, 1, 0, 0, 3.0, 10, 0], 10.0 / n, fps=FPS, record=rec_to(closing))
settle = []
st = p.run(st, [PI, 1, 0, 0, 6.0, LID, 0], [PI, 1, 0, 0, 6.0, LID, 0], 3.5, fps=FPS, record=rec_to(settle))
st = p.run(st, [PI, 1, 0, 0, 8.0, LID, 0], [PI, 1, 0, 0, 8.0, LID, 0], 1.5, fps=FPS, record=rec_to(settle))
stored = settle[-1]
print('guardada vmax', round(stored['vmax'], 3), 'z máx', round(float(stored['x'][:, 2].max()), 3), 'radio máx', round(float(np.hypot(stored['x'][:, 0] + 1, stored['x'][:, 1]).max()), 2),
      'deformación varilla', round(rod_strain(stored['x']), 4), 't', round(time.time() - t0), flush=True)
th_c = np.array([c['th'] for c in closing]); Qg_c = np.array([c['Qg'] for c in closing])
sel = th_c > np.deg2rad(97)
M0 = 1.20 * float((Qg_c[sel] / (-np.cos(th_c[sel] + p.delta))).max())
print('par máx al recoger', round(float(Qg_c.max()), 1), '→ M0', round(M0, 1), flush=True)
scale = M0 / p.M0
res = dict(open_pose=open_pose['x'], stored=stored['x'], lid=LID, M0=M0, delta=p.delta, sector_r=p.sector_r, th_open=thf, fps=FPS, mass=p.mass_total,
           d_per=d_per, loop=loop, cinch_nodes=np.asarray(p.cinch['nodes']), strain_stored=rod_strain(stored['x']),
           strain_max=max(rod_strain(c['x']) for c in (cinching + closing + settle)[::4]), Tc_hold=cinching[-1]['Tc'], Tc_max=max(c['Tc'] for c in cinching),
           open_Qg=open_pose['Qg'])
for name, lst in (('closing', closing), ('cinching', cinching)):
    res[f'{name}_x'] = np.stack([c['x'] for c in lst])
    for k in ('th', 'T', 'Qg', 'Ms', 'Tc'):
        res[f'{name}_{k}'] = np.array([c[k] for c in lst])
np.savez_compressed(out, **res)
print('guardado', round(time.time() - t0), 's')
