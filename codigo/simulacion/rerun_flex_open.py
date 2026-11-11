"""Reabre la opción B desde el estado guardado: voltea hasta TH_D, despliega ahí y luego baja al tope."""
import sys, time, numpy as np, jax.numpy as jnp
from configs import make_B, boundary_loop, nodes_on
from petal import rod
Z = dict(np.load('anim_flex.npz'))
TH_D = float(sys.argv[1]); out = sys.argv[2]
FPS, LID = 15, 2.0
d_per = int(Z['d_per'])
p, m = make_B(theta_open_deg=12.0)
AX = nodes_on(m['P'], y=0.0, lo=1.875, hi=4.0)
loop = boundary_loop(m); thf = float(p.th_open); PI = np.pi; thd = np.deg2rad(TH_D)
sm = lambda t: t * t * (3 - 2 * t)
def rec_to(lst):
    def f(state, mean, last):
        lst.append(dict(x=np.asarray(state[0], np.float32), th=float(state[2]), T=float(mean[0]), Qg=float(mean[1]), Ms=float(mean[2]), Tc=float(mean[4]), vmax=float(np.abs(np.asarray(state[1])).max())))
    return f
def rod_strain(x):
    r = x[np.r_[loop, loop[0], loop[1]]]; e = np.diff(r, axis=0); le = np.linalg.norm(e, axis=1); tt = e / le[:, None]
    ra = x[AX]; ea = np.diff(ra, axis=0); la = np.linalg.norm(ea, axis=1); ta = ea / la[:, None]
    ka = np.linalg.norm(np.diff(ta, axis=0), axis=1) / (0.5 * (la[:-1] + la[1:]))
    return float(max((np.linalg.norm(np.diff(tt, axis=0), axis=1) / (0.5 * (le[:-1] + le[1:]))).max(), ka.max()) * d_per / 2e3)
M0 = float(Z['M0']); scale = M0 / p.M0; r = p.sector_r
st = (jnp.asarray(Z['stored'], jnp.float64), jnp.zeros((p.N, 3)), jnp.asarray(PI), jnp.asarray(0.0))
opening = []; t0 = time.time()
n = 28
for k in range(n):                                   # 1. voltear contenido hasta TH_D
    a = PI - (PI - thd) * sm(k / n); b = PI - (PI - thd) * sm((k + 1) / n)
    la, lb = (LID + 0.3 * k, LID + 0.3 * (k + 1)) if k < 6 else (10, 10)
    st = p.run(st, [0, 0, r * (PI - a), scale, 2.0, la, 0], [0, 0, r * (PI - b), scale, 2.0, lb, 0], 7.0 / n, fps=FPS, record=rec_to(opening))
u_d = r * (PI - thd)
st = p.run(st, [0, 0, u_d, scale, 3.0, 10, 0], [0, 0, u_d, scale, 3.0, 10, 0], 0.6, fps=FPS, record=rec_to(opening))
n_flip = len(opening)
n = 24
for k in range(n):                                   # 2. soltar el cordón: el flex despliega
    a, b = sm(k / n), sm((k + 1) / n)
    st = p.run(st, [0, 0, u_d, scale, 2.5, 10, a], [0, 0, u_d, scale, 2.5, 10, b], 5.0 / n, fps=FPS, record=rec_to(opening))
st = p.run(st, [0, 0, u_d, scale, 3.0, 10, 1], [0, 0, u_d, scale, 3.0, 10, 1], 1.4, fps=FPS, record=rec_to(opening))
n_rel = len(opening)
n = 28
for k in range(n):                                   # 3. bajar desplegado hasta el tope
    a = thd - (thd - thf + 0.04) * sm(k / n); b = thd - (thd - thf + 0.04) * sm((k + 1) / n)
    st = p.run(st, [0, 0, r * (PI - a), scale, 2.0, 10, 1], [0, 0, r * (PI - b), scale, 2.0, 10, 1], 7.0 / n, fps=FPS, record=rec_to(opening))
u_end = r * (PI - thf + 0.04)
st = p.run(st, [0, 0, u_end, scale, 3.5, 10, 1], [0, 0, u_end, scale, 3.5, 10, 1], 3.5, fps=FPS, record=rec_to(opening))
x = opening[-1]['x']; P = m['P']; nA = np.array([-np.sin(thf), 0, np.cos(thf)]); h = x @ nA
tip = int(np.argmin(np.hypot(P[:, 0] - 4, P[:, 1])))
print('TH_D', TH_D, '| th final', round(np.rad2deg(opening[-1]['th']), 2), 'T máx', round(max(o['T'] for o in opening), 1), 'vmax fin', round(opening[-1]['vmax'], 3),
      '| punta h', round(float(-h[tip]), 2), 'lado h', round(float(-h[p.cinch['nodes'][1]]), 2), 'ancho', round(float(x[:, 1].max() - x[:, 1].min()), 2), 'alcance', round(float(x[:, 0].max()), 2),
      'deformación máx', round(max(rod_strain(o['x']) for o in opening[::3]), 4), round(time.time() - t0), 's', flush=True)
Z.update(final=x, n_flip=n_flip, n_rel=n_rel, th_deploy=thd)
Z['strain_max'] = max(float(Z['strain_max']), max(rod_strain(o['x']) for o in opening[::3]))
Z['opening_x'] = np.stack([c['x'] for c in opening])
for k in ('th', 'T', 'Qg', 'Ms', 'Tc'):
    Z[f'opening_{k}'] = np.array([c[k] for c in opening])
np.savez_compressed(out, **Z)
