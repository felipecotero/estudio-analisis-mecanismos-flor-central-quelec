import sys, json, time, numpy as np
from configs import make, boundary_loop
from petal import band, rod
mats = [('Varilla de fibra de vidrio Ø6 mm', rod(6), 6e-3, 0.01), ('Varilla de fibra de vidrio Ø8 mm', rod(8), 8e-3, 0.003), ('Fleje 1070 de 15×0,8 mm, de canto', band(15, 0.8), 0.8e-3, 0.01), ('Fleje 1070 de 15×2 mm, de canto', band(15, 2.0), 2e-3, 0.01)]
out = []
for label, mat, d, pre in mats:
    t0 = time.time()
    p, m = make('flex', perimeter=mat, antenna=rod(6), pre=pre, theta_open_deg=12.0)
    thf = float(p.th_open); P = m['P']; loop = boundary_loop(m)
    st = p.init_state(thf); c = [thf, 1, 0, 0, 6.0, 10, 1]; log = []
    st = p.run(st, c, c, 7.0, fps=10, record=lambda s, mean, last: log.append((float(np.abs(np.asarray(s[1])).max()), float(mean[1]))))
    x = np.asarray(st[0]); nA = np.array([-np.sin(thf), 0, np.cos(thf)]); h = x @ nA
    tip = int(np.argmin(np.hypot(P[:, 0] - 4, P[:, 1])))
    r = x[np.r_[loop, loop[0], loop[1]]]; e = np.diff(r, axis=0); le = np.linalg.norm(e, axis=1); tt = e / le[:, None]
    kap = np.linalg.norm(np.diff(tt, axis=0), axis=1) / (0.5 * (le[:-1] + le[1:]))
    row = dict(config='flex', label=label, mass=p.mass_total, vmax=log[-1][0], Qg=log[-1][1], tip_h=float(-h[tip]), side_h=float(-h[p.cinch['nodes'][1]]),
               span=float(x[:, 1].max() - x[:, 1].min()), reach=float(x[:, 0].max()), low_z=float(x[:, 2].min()), strain=float(kap.max() * d / 2), lam=mat['lam'], pre=pre, x=np.asarray(x, np.float32).tolist())
    out.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != 'x'}, round(time.time() - t0), 's', flush=True)
json.dump(out, open('poses_flex.json', 'w'))
