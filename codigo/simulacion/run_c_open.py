import numpy as np, jax.numpy as jnp, time
from sim_c import Flower, PETALS
f = Flower()
th = np.deg2rad([p['th_open'] for p in PETALS])
st = f.init_state(th); c = jnp.asarray(np.r_[th, np.ones(5), 6.0])
n = int(round(1 / 30 / f.dt)); rec = []
for k in range(30 * 7):
    K = f.build_pairs(st[0], np.asarray(st[2]))
    st, M, Qg, tc, non = f._chunk(st, c, c, f.pairs, n)
    rec.append((int(non), K))
x = np.asarray(f.eff(st[0], st[2]))
print('vmax', float(np.abs(np.asarray(st[1])).max()), 'contactos activos máx', max(r[0] for r in rec), 'M', np.round(np.asarray(M), 1), 'th', np.round(np.rad2deg(np.asarray(st[2])), 1))
np.savez('open_c.npz', x=x, th=np.asarray(st[2]), M=np.asarray(M), contacts=max(r[0] for r in rec))
