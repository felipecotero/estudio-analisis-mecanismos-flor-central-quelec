import time, sys, numpy as np, jax.numpy as jnp
from sim_c import Flower, PETALS
FPS = 10
f = Flower()
th_open = np.deg2rad([p['th_open'] for p in PETALS])
order = np.array([p['order'] for p in PETALS])          # 0 abre primero (queda arriba de la pila)
PI, TH_REL = np.pi, np.deg2rad(100)
sm = lambda u: (lambda c: c * c * (3 - 2 * c))(np.clip(u, 0, 1))
T_SETTLE, T_CINCH, T_FOLD, STAG_F = 3.0, 6.0, 8.0, 6.0
t_fold0 = T_SETTLE + T_CINCH + 1.0
t_fold_end = t_fold0 + 4 * STAG_F + T_FOLD + 3.0
STAG_O, T_FLIP, T_REL, T_LOW = 6.0, 6.0, 3.0, 4.0
t_open_end = t_fold_end + 4 * STAG_O + T_FLIP + 0.5 + T_REL + T_LOW + 2.5

def ctrl(t):
    th = th_open.copy(); cz = np.ones(5); damp = 3.0
    if t < T_SETTLE:
        damp = 6.0
    elif t < t_fold_end:
        cz[:] = 1 - sm((t - T_SETTLE) / T_CINCH)
        for i in range(5):
            ts = t_fold0 + (4 - order[i]) * STAG_F       # el último en abrir se recoge primero
            th[i] = th_open[i] + (PI - th_open[i]) * sm((t - ts) / T_FOLD)
    else:
        cz[:] = 0
        for i in range(5):
            ts = t_fold_end + order[i] * STAG_O          # el de arriba de la pila abre primero
            th[i] = PI + (TH_REL - PI) * sm((t - ts) / T_FLIP)
            t2 = ts + T_FLIP + 0.5
            cz[i] = sm((t - t2) / T_REL)
            th[i] = th[i] + (th_open[i] - TH_REL) * sm((t - t2 - T_REL) / T_LOW)
    return np.r_[th, cz, damp]

st = f.init_state(th_open)
n = int(round(1 / FPS / f.dt))
frames, rec = [], []
t, t0 = 0.0, time.time()
nf = int(round(t_open_end * FPS))
SUB = 3
n = int(round(1 / FPS / SUB / f.dt))
for k in range(nf):
    for j in range(SUB):
        K = f.build_pairs(st[0], np.asarray(st[2]))
        c0, c1 = jnp.asarray(ctrl(t)), jnp.asarray(ctrl(t + 1 / FPS / SUB))
        st, M, Qg, tc, non = f._chunk(st, c0, c1, f.pairs, n)
        t += 1 / FPS / SUB
    x = np.asarray(f.eff(st[0], st[2]), np.float32)
    frames.append(x)
    rec.append(np.r_[t, np.asarray(st[2]), np.asarray(M), np.asarray(Qg), np.asarray(tc), float(non), K, float(np.abs(np.asarray(st[1])).max())])
    if k % 20 == 0 or k == nf - 1:
        print(f't={t:5.1f}/{t_open_end:.1f} pares={K} activos={int(non)} vmax={rec[-1][-1]:.2f} th={np.round(np.rad2deg(np.asarray(st[2])),0)} M={np.round(np.asarray(M),0)} pared={time.time()-t0:.0f}s', flush=True)
    if not np.isfinite(x).all():
        print('NO FINITO'); break
    if k % 100 == 0 or k == nf - 1:
        np.savez_compressed('anim_c.npz', frames=np.stack(frames), rec=np.array(rec), fps=FPS, t_fold_end=t_fold_end, t_fold0=t_fold0,
                            order=order, th_open=th_open, offs=f.offs, STAG_O=STAG_O)
print('listo', round(time.time() - t0), 's')
