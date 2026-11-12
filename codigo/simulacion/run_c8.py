import time, numpy as np, jax.numpy as jnp
from sim_c8 import Flower8, PETALS
FPS = 10
f = Flower8()
th_open = np.deg2rad([p['th_open'] for p in PETALS]); order = np.array([p['order'] for p in PETALS])
sm = lambda u: (lambda c: c * c * (3 - 2 * c))(np.clip(u, 0, 1))
D100, D110 = np.deg2rad(100), np.deg2rad(110)
T0, STAG = 12.0, 3.0
T_END = T0 + 4 * STAG + 13 + 2
def ctrl(t):
    th = th_open.copy(); slack = np.zeros(5); w = np.zeros(5); tf = np.zeros(5); damp = 4.0
    if t < T0:                                               # plegado fuera de escena
        w[:] = sm(t / 0.5); slack[:] = 2 * sm(t / 1.0)
        th = th_open + (D100 - th_open) * sm((t - 0.5) / 3.5)
        tf[:] = sm((t - 4.0) / 6.0)
        th = th + (D110 - D100) * sm((t - 10.0) / 1.5)
        return np.r_[th, slack, w, damp], tf
    for i in range(5):
        ts = T0 + order[i] * STAG
        th[i] = D110 + (D100 - D110) * sm((t - ts) / 2.0)
        tf[i] = 1 - sm((t - ts - 2.0) / 6.0)
        u = sm((t - ts - 8.0) / 4.0)
        th[i] = th[i] + (th_open[i] - D100) * u
        slack[i] = 2 * (1 - u)
        w[i] = 1 - sm((t - ts - 12.0) / 1.0)
    return np.r_[th, slack, w, damp], tf
st = f.init_state(th_open)
n = int(round(1 / FPS / f.dt)); frames, rec = [], []
t, t0w = 0.0, time.time()
for k in range(int(round(T_END * FPS))):
    (c0, tf0), (c1, tf1) = ctrl(t), ctrl(t + 1 / FPS)
    tg0, tg1 = f.targets(tf0, c0[:5]), f.targets(tf1, c1[:5])
    st, M, T = f._chunk(st, jnp.asarray(c0), jnp.asarray(c1), (tg0, tg1), n)
    t += 1 / FPS
    if t >= T0 - 1e-6:
        frames.append(np.asarray(f.eff(st[0], st[2]), np.float32))
        rec.append(np.r_[t, np.asarray(st[2]), np.asarray(M), np.asarray(T), tf1, c1[10:15]])
    if k % 50 == 0:
        print(f't={t:5.1f}/{T_END} th={np.round(np.rad2deg(np.asarray(st[2])))} vmax={float(np.abs(np.asarray(st[1])).max()):.2f} M={np.round(np.asarray(M))} pared={time.time()-t0w:.0f}s', flush=True)
np.savez_compressed('anim_c8.npz', frames=np.stack(frames), rec=np.array(rec), fps=FPS, T0=T0, offs=f.offs)
print('listo', round(time.time() - t0w))
