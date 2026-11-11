"""Posturas abiertas en equilibrio para comparar materiales del aro y el apoyo del vuelo."""
import sys, json, time
import numpy as np
import jax.numpy as jnp
from configs import make, nodes_on
from petal import band, rod, GFRP, tube_alu

TH = 12.0
jobs = {
    'A': [('literal', band(15, 0.5), None), ('literal', band(15, 0.8), None), ('literal', band(15, 1.0), None),
          ('literal', rod(6), None), ('literal', rod(8), None), ('antena', band(15, 0.8), rod(8))],
    'B': [('cruceta', band(15, 0.5), None), ('cruceta', band(15, 0.8), None), ('cruceta', band(15, 1.0), None),
          ('cruceta', rod(6), None), ('cruceta', rod(8), None), ('antena', band(15, 0.8), rod(10))],
    'C': [('literal', tube_alu(19, 1.5), None), ('literal', band(15, 0.5), None)],
}[sys.argv[1]]
out = []
for cfg, hoop, ant in jobs:
    t0 = time.time()
    kw = dict(antenna=ant) if ant else {}
    if sys.argv[1] == 'C':
        kw['dt'] = 5e-5
    p, m = make(cfg, hoop=hoop, theta_open_deg=TH, **kw)
    thf = float(p.th_open)
    st = p.init_state(thf)
    log = []
    c = [thf, 1, 0, 0, 6.0, 10]
    st = p.run(st, c, c, 6.0, fps=10, record=lambda s, mean, last: log.append((float(np.abs(np.asarray(s[1])).max()), float(mean[1]))))
    x = np.asarray(st[0]); P = m['P']; nh = p.nh
    nA = np.array([-np.sin(thf), 0, np.cos(thf)])
    hgt = x @ nA                                   # altura respecto al plano ideal del pétalo
    tip = int(np.argmin(np.hypot(P[:, 0] - 4, P[:, 1])))
    ki, ko, tw = [np.asarray(a) for a in p.hoop_diag(jnp.asarray(x), jnp.asarray(thf))]
    if hoop['kind'] == 'fleje':
        b, t = hoop['b_mm'] / 1e3, hoop['t_mm'] / 1e3
        sig = hoop['E'] * (np.abs(ki) * t / 2 + np.abs(ko) * b / 2); tau = 80e9 * t * np.abs(tw)
    else:
        d = hoop['d_mm'] / 1e3
        sig = hoop['E'] * np.hypot(ki, ko) * d / 2; tau = (26e9 if hoop['kind'] == 'tubo' else 4e9) * d / 2 * np.abs(tw)
    row = dict(config=cfg, hoop=hoop['label'], antenna=ant['label'] if ant else None, mass=p.mass_total, vmax=log[-1][0], Qg=log[-1][1],
               far_drop=float(-hgt[nh // 2]), side_drop=float(-0.5 * (hgt[nh // 4] + hgt[3 * nh // 4])), disc_sag=float(-hgt[np.asarray(p.disc_idx)].min()),
               tip=x[tip].tolist(), tip_drop=float(4 * np.sin(thf) - x[tip, 2]), low_z=float(x[:, 2].min()), reach=float(x[:, 0].max()),
               span=float(x[:, 1].max() - x[:, 1].min()), sigma=float(sig.max() / 1e6), tau=float(tau.max() / 1e6))
    if ant:
        idx = nodes_on(P, y=0.0, lo=1.875, hi=4.0)
        r = x[idx]; e = np.diff(r, axis=0); le = np.linalg.norm(e, axis=1); tt = e / le[:, None]
        kap = np.linalg.norm(np.diff(tt, axis=0), axis=1) / (0.5 * (le[:-1] + le[1:]))
        row['antenna_sigma'] = float(GFRP['E'] * kap.max() * ant['d_mm'] / 2e3 / 1e6)
    row['x'] = np.asarray(x, np.float32).tolist()
    out.append(row)
    print(cfg, hoop['label'], ant['label'] if ant else '', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k not in ('x', 'tip')}, round(time.time() - t0), 's', flush=True)
json.dump(out, open(f'poses_{sys.argv[1]}.json', 'w'))
