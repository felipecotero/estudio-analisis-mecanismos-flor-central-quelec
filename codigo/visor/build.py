"""Empaqueta la simulación y el visor en un HTML autónomo."""
import base64, json, subprocess, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/claude/work/sim')
import os; os.chdir('/home/claude/work/sim')
import mesh as M
from configs import nodes_on, boundary_loop

W = Path('/home/claude/work')
SRC = Path('/mnt/user-data/uploads/24 QUELEC Y LA SEMILLA DEL FUTURO - MARIO AVILA/output/Flor-central-Quelec/03-Visor-3D/Codigo')
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
mm = lambda x: np.round(np.asarray(x, np.float64) * 1000).clip(-32000, 32000).astype('<i2')
es = lambda v, d=0: f'{v:,.{d}f}'.replace(',', '§').replace('.', ',').replace('§', '.')


def smooth(v, w):
    pad = np.r_[np.full(w // 2, v[0]), v, np.full(w // 2, v[-1])]
    return np.convolve(pad, np.ones(w) / w, mode='valid')


mA = M.build(h=0.125)
mB = M.build(h=0.125, r_tip=0.5)
PA, PB = mA['P'], mB['P']
A = np.load(W / 'sim/anim_cruceta.npz')
B = np.load(W / 'sim/anim_flex2.npz')
loopB = boundary_loop(mB)
perim = float(np.hypot(*(PB[np.r_[loopB[1:], loopB[0]]] - PB[loopB]).T).sum())
geom = dict(ringR=1.0, z0=0.24, pitch=0.06, lowerZ=0.06, platform=[2.31, 1.20], sectorR=float(A['sector_r']), fair=[-0.05, -0.20],
            pitchB=0.12, anchorA=0.20, hornB=0.16, deltaDeg=float(np.rad2deg(A['delta'])), stage=dict(z=2.75, size=9.0, trapA=2.6, trapB=4.1, trapC=2.6, trapC8=2.6), liftStowed=1.40, liftTop=9.75)


def mesh_json(m, ribs):
    return dict(N=len(m['P']), nHoop=m['n_hoop'], P=b64(m['P'].astype('<f4')), tri=b64(m['tri'].astype('<u2')), length=4.0, width=3.5, ribs=ribs)


meshes = dict(
    A=mesh_json(mA, dict(spreader=nodes_on(PA, s=1.0, lo=-1.0, hi=1.0).tolist(), antenna=nodes_on(PA, y=0.0, lo=1.875, hi=4.0).tolist(), perimeter=[], cinch=[])),
    B=mesh_json(mB, dict(spreader=nodes_on(PB, s=1.0, lo=-1.0, hi=1.0).tolist(), antenna=nodes_on(PB, y=0.0, lo=1.875, hi=4.0).tolist(), perimeter=loopB.tolist(),
                          cinch=B['cinch_nodes'].tolist(), cinchAnchors=[[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]])),
)


def anim_json(Z, key, extra=None):
    T_s = smooth(Z['opening_T'], 9)          # media móvil de 0,6 s: tensión casi estática, sin los rebotes del arranque
    Qg_s = smooth(Z['opening_Qg'], 5)
    d = dict(mesh=key, nF=len(Z['opening_x']), fps=int(Z['fps']), frames=b64(mm(Z['opening_x'])), th=Z['opening_th'].round(4).tolist(), T=T_s.round(2).tolist(),
             Qg=Qg_s.round(2).tolist(), Ms=Z['opening_Ms'].round(2).tolist(), thOpen=float(Z['th_open']))
    d.update(extra or {})
    return d, T_s, Qg_s


n_flip = int(B['n_flip'])
animA, TA, QA = anim_json(A, 'A')
animB, TB, QB = anim_json(B, 'B', dict(nFlip=n_flip, nCinch=int(B['n_rel']) - 15, Tc=smooth(B['opening_Tc'], 5).round(2).tolist()))

# ---------- posturas
poses_raw = []
for k in 'ABC':
    f = W / f'sim/poses_{k}.json'
    if f.exists():
        poses_raw += json.load(open(f))
poses, seen = [], set()
order = ['Fleje 1070 15×0.5 mm', 'Fleje 1070 15×0.8 mm', 'Fleje 1070 15×1 mm', 'Varilla fibra de vidrio Ø6 mm', 'Varilla fibra de vidrio Ø8 mm', 'Tubo de aluminio Ø19×1.5 mm rolado']
nice = lambda s: s.replace('.', ',').replace('Fleje 1070', 'Fleje 1070 de').replace(' rolado', ', rolado')
for r in reversed(poses_raw):                         # la última corrida de cada caso manda
    key = (r['config'], r['hoop'], r['antenna'])
    if key in seen or not np.isfinite(r['vmax']) or r['vmax'] > 5:
        continue
    seen.add(key)
    poses.append(r)
poses.sort(key=lambda r: (r['config'], order.index(r['hoop']), r['antenna'] or ''))
out_poses = []
for r in poses:
    if r['config'] == 'cruceta':
        continue
    label = nice(r['hoop']) if r['config'] == 'literal' else nice(r['antenna'])
    out_poses.append(dict(id=f"{r['config']}-{len(out_poses)}", mesh='A', config=r['config'], label=label, x=b64(mm(r['x'])), mass=r['mass'], Qg=r['Qg'],
                          far_drop=r['far_drop'], tip_drop=r['tip_drop'], low_rel=r['low_z'], reach=r['reach'], antenna_sigma=r.get('antenna_sigma', 0),
                          default=(r['hoop'] == order[1] and (r['antenna'] in (None, 'Varilla fibra de vidrio Ø8 mm')))))
flex_poses = json.load(open(W / 'sim/poses_flex.json'))
for k, r in enumerate(flex_poses):
    out_poses.append(dict(id=f'flex-{k}', mesh='B', config='flex', label=r['label'], x=b64(mm(r['x'])), mass=r['mass'], Qg=r['Qg'], tip_h=r['tip_h'], side_h=r['side_h'],
                          span=r['span'], reach=r['reach'], default=(k == 0)))

tex = ['data:image/webp;base64,' + base64.b64encode((SRC / f'flor-central/artes-preview/FG0{i}.webp').read_bytes()).decode() for i in range(1, 6)]
# ---------- opción C
import jax.numpy as jnp
import mesh_c as MC
from sim_c import Flower, PETALS
FC = Flower()
src = open(W / 'sim/run_c.py').read()
src = src[:src.index('st = f.init_state')].replace('f = Flower()', 'f = FC')
ns = dict(FC=FC); exec(src, ns)
Co = np.load(W / 'sim/open_c.npz')
gC = np.asarray(FC.to_global(jnp.asarray(Co['x'])))
def cmesh(big):
    m = FC.meshes[big]
    return dict(N=len(m['P']), nHoop=0, P=b64(m['P'].astype('<f4')), tri=b64(m['tri'].astype('<u2')), length=float(m['L']), width=float(m['width']),
                ribs=dict(spreader=m['spreader'].tolist(), antenna=m['antenna'].tolist(), perimeter=m['perimeter'].tolist(), cinch=m['cinch'].tolist(),
                          cinchAnchors=[[1.0, 0.9], [1.0, 0.0], [1.0, 0.0]]))
meshes['Cn'] = cmesh(False); meshes['Cb'] = cmesh(True)
mass = [float(FC.m_true_np[FC.offs[k]:FC.offs[k + 1]].sum()) for k in range(5)]
low = [float(0.15 - gC[FC.pid_np == k, 2].min()) for k in range(5)]
NC = dict(mass=mass, low=low, th=np.rad2deg(Co['th']).tolist(), M=Co['M'].tolist(), open_r=float(np.hypot(gC[:, 0], gC[:, 1]).max()), cross=int(np.load(W / 'sim/cross_open_c.npy')[0]))
json.dump(NC, open(W / 'viewer/numbers_c.json', 'w'), indent=1)
CINFO = dict(az=[p['az'] for p in PETALS], tex=[p['tex'] for p in PETALS], mesh=['Cb' if p['big'] else 'Cn' for p in PETALS], hingeZ=0.15,
             N=int(FC.N), offs=[int(o) for o in FC.offs[:5]], x=b64(mm(Co['x'])), th=[float(v) for v in Co['th']])
names = [p['name'] for p in PETALS]
rows_c = ''.join(f"<tr><td>{names[k]}</td><td class=\"num\">{es(NC['th'][k], 0)}°</td><td class=\"num\">{es(NC['M'][k], 0)} N·m</td><td class=\"num\">{es(NC['low'][k], 2)} m</td><td class=\"num\">{es(mass[k], 1)} kg</td></tr>" for k in range(5))
numbers_c = f'''
<h3>La flor C abierta</h3>
<table><tr><th>Pétalo</th><th>Ángulo</th><th>Par en la bisagra</th><th>Cae bajo la bisagra</th><th>Masa</th></tr>{rows_c}</table>
<table>
<tr><td>Flor abierta: radio desde el centro</td><td class="num">{es(NC['open_r'], 2)} m</td></tr>
<tr><td>Cruces de tela en la postura abierta</td><td class="num">{NC['cross']}</td></tr>
</table>
<p class="small">Abiertos, los pétalos no llegan a tocarse: el contacto no tuvo que actuar en esta postura. Cada uno abre a un ángulo distinto para que, al bajar uno después de otro, el que llega después quede encima.</p>
<h3>Lo que se intentó al recogerla y por qué falló</h3>
<ul>
<li><strong>Lámina entera doblada sobre el centro,</strong> como en el modelo de Cinema 4D. Cada lámina mide 3,25 m desde el aro y llega a 2,25 m del centro por el otro lado. Queda encima de la base de los pétalos que aún no se doblan: ya no pueden subir sin atravesarla. Con cinco pétalos de este largo y un aro de 1 m, no hay un orden que lo evite.</li>
<li><strong>Lámina contenida con el cordón y luego doblada.</strong> La tela se arruga sobre sí misma. Este cálculo detecta el choque de un punto contra una cara, pero no alcanza a impedir los cruces de la tela arrugada: cientos consigo misma y decenas con las vecinas. Además, dos pétalos se frenaron a 120–160° sobre la pila, con el mando a 100 N·m.</li>
<li><strong>Qué haría falta.</strong> O un plegado distinto, que reduzca cada lámina a menos de 1 m antes de voltearla (el plegado en ocho de los reflectores, por ejemplo), o un cálculo de contacto de otra categoría, más costoso. Lo más útil ahora es probarlo con un pétalo de tela y varilla a escala real.</li>
</ul>'''
# ---------- opción C plegada en ocho
from sim_c8 import Flower8, MAST_H
F8 = Flower8()
A8 = np.load(W / 'sim/anim_c8.npz'); Fr8 = A8['frames']; rec8 = A8['rec']; n8 = len(Fr8)
th8 = rec8[:, 1:6]; Mh8 = rec8[:, 6:11]; T8 = rec8[:, 11:26]; tf8 = rec8[:, 26:31]; w8 = rec8[:, 31:36]
cr8 = np.load(W / 'sim/cross_c8.npy')
open8 = ((tf8 < 0.02) & (th8 < np.deg2rad(40))).sum(1)
anim8 = dict(N=int(F8.N), offs=[int(o) for o in F8.offs[:5]], nF=n8, fps=int(A8['fps']), frames=b64(mm(Fr8)), th=th8.round(4).ravel().tolist(),
             tf=tf8.round(3).ravel().tolist(), open=open8.tolist(), Tmax=T8.max(1).round(1).tolist(), crossAll=cr8[:, 1].astype(int).tolist(), crossP=cr8[:, 2].astype(int).tolist())
g8s = np.asarray(F8.to_global(jnp.asarray(Fr8[0], float))); g8o = np.asarray(F8.to_global(jnp.asarray(Fr8[-1], float)))
N8 = dict(mass=[float(F8.m_true_np[F8.offs[k]:F8.offs[k + 1]].sum()) for k in range(5)], coilR=[float(r) for r in F8.coilR], loopL=[float(l) for l in F8.loopL],
          strain=float(F8.rod['d'] / 2 / min(F8.coilR)), bud_r=float(np.hypot(g8s[:, 0], g8s[:, 1]).max()), bud_h=float(g8s[:, 2].max()),
          open_r=float(np.hypot(g8o[:, 0], g8o[:, 1]).max()), Tmax=float(T8[-20:].max()), Tmax_all=float(T8.max()), Mmax=float(np.abs(Mh8).max()),
          crossP_max=int(cr8[:, 2].max()), cross_end=int(cr8[-1, 1]), cross_stored=int(cr8[0, 1]))
json.dump(N8, open(W / 'viewer/numbers_c8.json', 'w'), indent=1)
numbers_c8 = f'''
<h3>Un pétalo C plegado en ocho</h3>
<table>
<tr><td>Varilla de contorno (fibra de vidrio Ø10 mm), una pieza</td><td class="num">{es(N8['loopL'][1], 1)} m · {es(N8['loopL'][0], 1)} m el grande</td></tr>
<tr><td>Diámetro de los tres aros plegados</td><td class="num">{es(2 * N8['coilR'][1], 2)} m · {es(2 * N8['coilR'][0], 2)} m el grande</td></tr>
<tr><td>Deformación de la varilla plegada</td><td class="num">{es(N8['strain'] * 100, 1)} %</td></tr>
<tr><td>Masa de un pétalo / del grande</td><td class="num">{es(N8['mass'][1], 1)} kg · {es(N8['mass'][0], 1)} kg</td></tr>
<tr><td>Tensor más cargado, abierta</td><td class="num">{es(N8['Tmax'], 0)} N</td></tr>
<tr><td>Par máximo en una bisagra</td><td class="num">{es(N8['Mmax'], 0)} N·m</td></tr>
</table>
<h3>Guardada y abierta</h3>
<table>
<tr><td>Capullo guardado: diámetro</td><td class="num">{es(2 * N8['bud_r'], 2)} m</td></tr>
<tr><td>Capullo guardado: alto sobre la plataforma</td><td class="num">{es(N8['bud_h'], 2)} m</td></tr>
<tr><td>Mástil central sobre la bisagra</td><td class="num">{es(MAST_H, 1)} m</td></tr>
<tr><td>Flor abierta: radio desde el centro</td><td class="num">{es(N8['open_r'], 2)} m</td></tr>
</table>
<h3>Choques pasados en alto</h3>
<table>
<tr><td>Cruces de tela guardada (casi todos de cada tela consigo misma)</td><td class="num">{N8['cross_stored']}</td></tr>
<tr><td>Máximo de cruces entre pétalos distintos al abrir</td><td class="num">{N8['crossP_max']}</td></tr>
<tr><td>Cruces con la flor abierta</td><td class="num">{N8['cross_end']}</td></tr>
</table>
<h3>Cómo se arma</h3>
<ul>
<li><strong>Una varilla por pétalo,</strong> de fibra de vidrio de 10 mm, en una funda por todo el contorno de la lámina, con los dos extremos fijos en la bisagra del aro. No hay brazo rígido ni antena: estorbarían el plegado.</li>
<li><strong>Plegado en ocho fuera de escena:</strong> se tuerce la mitad lejana, se forma el ocho y se recoge en tres aros. Una funda o una correa con liberación lo retiene; soltarla de forma dosificada evita que el aro salte de golpe.</li>
<li><strong>Mástil central de {es(MAST_H, 1)} m</strong> con tres tensores a cada pétalo (punta y dos costados). Están flojos mientras el pétalo sube y se despliega, y lo toman al bajar.</li>
<li><strong>Límite de la varilla.</strong> Plegada trabaja al {es(N8['strain'] * 100, 1)} %, cerca del límite de la fibra de vidrio. Si el prototipo muestra fatiga, bajar a 9 mm o agrandar los aros.</li>
<li><strong>Riesgo de seguridad:</strong> un aro de resorte de 10 m que se suelta de golpe latiguea. La liberación tiene que ser controlada.</li>
</ul>'''
CINFO['mastH'] = MAST_H
data = dict(meshes=meshes, anims=dict(cruceta=animA, flex=animB, c8=anim8), poses=out_poses, geom=geom, textures=tex, textureOrder=[2, 0, 1, 3, 4], C=CINFO)

# ---------- números de la opción A
a, b = geom['anchorA'], geom['hornB']
stag = 4.0


def common(Z, T, Qg, P):
    nF = len(T); fps = int(Z['fps'])
    tt = np.arange(0, 4 * stag + nF / fps, 1 / fps)
    tot = sum(np.interp(tt - k * stag, np.arange(nF) / fps, T, left=0, right=0) for k in range(5))
    M0 = float(Z['M0']); kel = M0 / (a * b)
    thf = float(Z['th_open'])
    d_max = max(np.sqrt(a * a + b * b + 2 * a * b * np.sin(t + np.deg2rad(180 + geom['deltaDeg']))) for t in np.linspace(thf, np.pi, 200))
    st = Z['stored']
    T = np.r_[np.zeros(15), T[15:]]                    # sin el rebote del primer segundo
    tot = sum(np.interp(tt - k * stag, np.arange(nF) / fps, T, left=0, right=0) for k in range(5))
    return dict(mass=float(Z['mass']), Qclose=float(Z['closing_Qg'].max()), M0=M0, kel=kel, Fel=kel * d_max, Tmax=float(T.max()), Ttot=float(tot.max()),
                travel=geom['sectorR'] * (np.pi - thf), pk_h=geom['z0'] + 4 * geom['pitch'] + float(st[:, 2].max()),
                pk_r=max(float(np.hypot(st[:, 0] + geom['ringR'], st[:, 1]).max()), geom['ringR'] + b * np.cos(np.deg2rad(geom['deltaDeg']))),
                hold=float(-(Qg[-1] + Z['opening_Ms'][-1])), thf=thf)


NA = common(A, TA, QA, PA)
op = A['open_pose']; thf = NA['thf']; nA = np.array([-np.sin(thf), 0, np.cos(thf)])
rdA = np.hypot(PA[:, 0] - 1, PA[:, 1])
NA.update(Qopen=float(-A['closing_Qg'].min()), sigma=float(A['hoop_sigma_max_MPa']), disc_sag=float(-(op[rdA < 1] @ nA).min()), low=float(op[:, 2].min()),
          diam=2 * (geom['ringR'] + float(op[:, 0].max())))
NB = common(B, TB, QB, PB)
fin = B['final']; hB = fin @ nA
tipB = int(np.argmin(np.hypot(PB[:, 0] - 4, PB[:, 1]))); sideB = int(B['cinch_nodes'][1])
NB.update(Qopen=float(-QB[-1]), strain=float(B['strain_max']), Tc_hold=float(B['Tc_hold']), Tc_max=float(B['Tc_max']), tip_h=float(-hB[tipB]), side_h=float(-hB[sideB]),
          span=float(fin[:, 1].max() - fin[:, 1].min()), reach=float(fin[:, 0].max()), diam=2 * (geom['ringR'] + float(fin[:, 0].max())), perim=perim, area=float(mB['total_area']),
          d_per=int(B['d_per']), bundle_h=float(B['stored'][:, 2].max()), flat_tip=flex_poses[0]['tip_h'], flat_side=flex_poses[0]['side_h'])
json.dump(dict(A=NA, B=NB), open(W / 'viewer/numbers.json', 'w'), indent=1)
ud = lambda v: (f'{es(v, 2)} m abajo' if v >= 0 else f'{es(-v, 2)} m arriba')

numbers = f'''
<h3>Un pétalo A2: aro + cruceta (fleje de 15 × 0,8 mm)</h3>
<table>
<tr><td>Masa: aro, tela, brazo, travesaño y raíz</td><td class="num">{es(NA['mass'], 1)} kg</td></tr>
<tr><td>Par del peso al levantar desde la pila</td><td class="num">{es(NA['Qclose'], 0)} N·m</td></tr>
<tr><td>Par del peso ya abierto</td><td class="num">{es(NA['Qopen'], 0)} N·m</td></tr>
<tr><td>Elástico compensador: par máximo que debe dar</td><td class="num">{es(NA['M0'], 0)} N·m</td></tr>
<tr><td>Rienda: tensión máxima de un pétalo</td><td class="num">{es(NA['Tmax'], 0)} N</td></tr>
<tr><td>Rienda: las cinco a un solo mando</td><td class="num">{es(NA['Ttot'], 0)} N</td></tr>
<tr><td>Rienda: recorrido por pétalo</td><td class="num">{es(NA['travel'] * 100, 0)} cm</td></tr>
<tr><td>Esfuerzo máximo en el fleje</td><td class="num">{es(NA['sigma'], 0)} MPa</td></tr>
<tr><td>Tela del disco: bolsa por su peso</td><td class="num">{es(NA['disc_sag'] * 100, 0)} cm</td></tr>
<tr><td>Vuelo sin apoyo: cuelga bajo la raíz</td><td class="num">{es(-NA['low'], 2)} m</td></tr>
</table>
<h3>El paquete guardado</h3>
<table>
<tr><td>Diámetro, con las palancas del elástico</td><td class="num">{es(2 * NA['pk_r'], 2)} m</td></tr>
<tr><td>Alto sobre el piso de la plataforma</td><td class="num">{es(NA['pk_h'], 2)} m</td></tr>
<tr><td>Flor abierta, de borde a borde</td><td class="num">{es(NA['diam'], 1)} m</td></tr>
</table>
<h3>Cómo se arma el mecanismo</h3>
<ul>
<li><strong>Raíces escalonadas.</strong> Cada bisagra va {es(geom['pitch'] * 100, 0)} cm más alta que la anterior, para que los cinco aros se apilen planos sobre el centro.</li>
<li><strong>Sector de {es(geom['sectorR'] * 200, 0)} cm de diámetro</strong> solidario a la raíz. La rienda se arrolla en él: su brazo de palanca no cambia en todo el giro.</li>
<li><strong>Elástico compensador.</strong> Caucho de la palanca ({es(b * 100, 0)} cm) a un anclaje {es(a * 100, 0)} cm bajo la bisagra. Con esas medidas pide unos {es(round(NA['kel'], -1), 0)} N/m y llega a {es(round(NA['Fel'], -1), 0)} N de tirón. El caucho se elige y se afina en la prueba.</li>
<li><strong>Rienda.</strong> Del sector baja al aro inferior, entra al tubo central y llega al mando bajo la tarima. Soltarla abre; halarla recoge para la función siguiente.</li>
<li><strong>Tope y pestillo.</strong> El tope fija el pétalo a 12°. Abierto, solo lo sujetan {es(NA['hold'], 0)} N·m, y una brisa desde abajo puede levantarlo. Necesita pestillo.</li>
</ul>'''

numbers_b = f'''
<h3>Un pétalo B: cruceta + antena + flex de contorno (varilla de {NB['d_per']} mm)</h3>
<table>
<tr><td>Masa: lo de A2 más antena, flex y más tela</td><td class="num">{es(NB['mass'], 1)} kg</td></tr>
<tr><td>Varilla del contorno: largo de una pieza</td><td class="num">{es(NB['perim'] * 1.01, 1)} m</td></tr>
<tr><td>Tela por pétalo</td><td class="num">{es(NB['area'], 1)} m²</td></tr>
<tr><td>Par del peso al levantar, contenido</td><td class="num">{es(NB['Qclose'], 0)} N·m</td></tr>
<tr><td>Par del peso abierto y desplegado</td><td class="num">{es(NB['Qopen'], 0)} N·m</td></tr>
<tr><td>Elástico compensador: par máximo que debe dar</td><td class="num">{es(NB['M0'], 0)} N·m</td></tr>
<tr><td>Rienda: tensión máxima de un pétalo</td><td class="num">{es(NB['Tmax'], 0)} N</td></tr>
<tr><td>Rienda: las cinco a un solo mando</td><td class="num">{es(NB['Ttot'], 0)} N</td></tr>
<tr><td>Cordón de contención: tensión por ramal</td><td class="num">{es(NB['Tc_max'], 0)} N</td></tr>
<tr><td>Varilla: deformación máxima en el recorrido</td><td class="num">{es(NB['strain'] * 100, 1)} %</td></tr>
</table>
<h3>Abierto y guardado</h3>
<table>
<tr><td>Punta tras desplegar, respecto al plano</td><td class="num">{ud(NB['tip_h'])}</td></tr>
<tr><td>Lados tras desplegar</td><td class="num">{ud(NB['side_h'])}</td></tr>
<tr><td>Ancho visible</td><td class="num">{es(NB['span'], 2)} m</td></tr>
<tr><td>Flor abierta, de borde a borde</td><td class="num">{es(NB['diam'], 1)} m</td></tr>
<tr><td>Paquete guardado: diámetro</td><td class="num">{es(2 * NB['pk_r'], 2)} m</td></tr>
<tr><td>Alto de un pétalo contenido, sin aplastar</td><td class="num">{es(NB['bundle_h'], 2)} m</td></tr>
</table>
<h3>Cómo se arma el mecanismo</h3>
<ul>
<li><strong>La base es A2:</strong> bisagras escalonadas, brazo de aluminio, travesaño, sector, elástico compensador, rienda, tope y pestillo.</li>
<li><strong>Antena de punta.</strong> Varilla de 2 m que prolonga el brazo hasta la punta. Sin ella la punta cuelga casi 2 m: el flex solo no la sostiene.</li>
<li><strong>Flex de contorno.</strong> Una sola varilla en una funda por todo el borde, un 1 % más larga que el borde para que empuje la tela. Se cierra con un casquillo junto a la raíz.</li>
<li><strong>Punta redonda.</strong> La varilla no puede hacer un pico: la punta lleva un arco de 50 cm de radio.</li>
<li><strong>Cordón de contención.</strong> Tres ramales por el envés. Los dos de los lados van al cruce del brazo y el travesaño. El de la punta va a un costado del aro, para que la antena se curve de lado y no se quiebre.</li>
<li><strong>Orden.</strong> El pétalo voltea contenido hasta unos 100°. Ahí se suelta el cordón y el flex abre hacia arriba. Después baja abierto hasta el tope.</li>
<li><strong>Dos posturas estables.</strong> Armado en plano, el pétalo queda con la punta {es(NB['flat_tip'], 2)} m abajo y los lados {es(NB['flat_side'], 2)} m abajo. Tras desplegar queda con la punta más caída y los lados a nivel. La prueba dirá cuál sale.</li>
<li><strong>Límite de la varilla.</strong> La fibra de vidrio aguanta cerca del 1 % de deformación sostenida y aquí llega a eso. Si la prueba lo confirma, usar varilla de 5 mm o lazos más amplios.</li>
<li><strong>Paquete.</strong> Los lazos de capas vecinas se cruzan en el dibujo; en la realidad se entrelazan. El alto real de la pila hay que medirlo en la prueba.</li>
</ul>'''

rows = {}
for r in poses:
    rows.setdefault(r['hoop'], {})[r['config']] = r
tbl = '<tr><th>Aro</th><th>Solo, sujeto en la raíz</th><th>Con cruceta</th><th>Masa del aro</th></tr>'
lam = {'Fleje 1070 15×0.5 mm': 7850 * 15e-3 * 0.5e-3, 'Fleje 1070 15×0.8 mm': 7850 * 15e-3 * 0.8e-3, 'Fleje 1070 15×1 mm': 7850 * 15e-3 * 1e-3,
       'Varilla fibra de vidrio Ø6 mm': 1900 * np.pi * 0.003 ** 2, 'Varilla fibra de vidrio Ø8 mm': 1900 * np.pi * 0.004 ** 2,
       'Tubo de aluminio Ø19×1.5 mm rolado': 2700 * np.pi * (0.019 ** 2 - 0.016 ** 2) / 4}
for h in order:
    if h not in rows:
        continue
    lit, cru = rows[h].get('literal'), rows[h].get('cruceta')
    c1 = 'Sin dato: el cálculo no convergió' if lit is None else (f"Cuelga: cae {es(lit['far_drop'], 2)} m" if lit['far_drop'] > 0.6 else f"Cede: cae {es(lit['far_drop'] * 100, 0)} cm")
    c2 = 'No calculado' if cru is None else f"Lados caen {es(cru['side_drop'] * 100, 0)} cm"
    tbl += f"<tr><td>{nice(h)}</td><td>{c1}</td><td>{c2}</td><td class=\"num\">{es(lam[h] * 2 * np.pi, 2)} kg</td></tr>"
tbl_b = '<tr><th>Flex de contorno</th><th>Punta</th><th>Lados</th><th>Ancho</th><th>Masa del flex</th></tr>'
for r in flex_poses:
    verdict = 'Cuelga: ' if r['tip_h'] > 1.0 else ''
    tbl_b += f"<tr><td>{r['label']}</td><td>{verdict}{ud(r['tip_h'])}</td><td>{ud(r['side_h'])}</td><td class=\"num\">{es(r['span'], 2)} m</td><td class=\"num\">{es(r['lam'] * perim, 2)} kg</td></tr>"

scope = f'''
<li><strong>Sí calcula:</strong> peso real de tela, aro y varillas; rigidez del fleje de canto en sus dos ejes y a torsión; flexión de las varillas; tela que se arruga y no atraviesa el disco; bisagra con elástico, tope y rienda; cordón de contención.</li>
<li><strong>Un pétalo, cinco copias.</strong> El choque entre pétalos vecinos y entre capas de la pila no se calcula. En la opción B los pétalos abiertos se montan unos sobre otros unos 25 cm: en la realidad se apoyan, aquí se cruzan.</li>
<li><strong>La varilla puede cruzarse consigo misma</strong> al recogerse: el cálculo no lo impide. Los tres lazos reales hay que acomodarlos a mano en la prueba.</li>
<li><strong>Sin viento.</strong> Cinco pétalos suman entre 45 y 49 m² de tela a 10 m. El manual de la Compact 12 admite 20 daN de empuje lateral en exterior. Hay que resolverlo con el proveedor de la máquina antes de subirla.</li>
<li><strong>Brazo rígido.</strong> El tubo de aluminio de 25 mm se toma como indeformable; real, su punta cede unos centímetros.</li>
<li><strong>Fleje como flex de contorno:</strong> se calculó solo con su eje débil. Es una simplificación que lo desfavorece un poco.</li>
<li><strong>Tela supuesta:</strong> 80 g/m². Con una tela más pesada suben todos los pares en proporción.</li>
<li><strong>Medidas supuestas:</strong> bastidor de 2 m de diámetro sobre la plataforma de 2,31 × 1,20 m; piso guardado a 1,40 m; tarima a 2,75 m; boca de 2,6 m para la opción A y de 4,1 m para la B. Cámbialas cuando midan la máquina y la trampa.</li>
<li><strong>Fuera de este estudio:</strong> el tallo, las hojas de bore y la compuerta de la trampa.</li>'''

subprocess.run(['npx', 'esbuild', 'src/main.js', '--bundle', '--format=iife', '--minify', '--outfile=bundle.js', '--log-level=warning'], cwd=W / 'viewer', check=True)
fonts = ''
for wgt, name in ((400, 'Regular'), (600, 'SemiBold'), (700, 'Bold')):
    fb = base64.b64encode((SRC / f'assets/fonts/Montserrat-{name}.woff2').read_bytes()).decode()
    fonts += f'@font-face{{font-family:Montserrat;src:url(data:font/woff2;base64,{fb}) format("woff2");font-weight:{wgt};font-display:swap}}\n'
html = (W / 'viewer/template.html').read_text()
html = html.replace('/*FONTS*/', fonts)
for old, new in (('<div id="numbers" data-show="optA"></div>', '<div id="numbers" data-show="optA">' + numbers + '</div>'),
                 ('<div id="numbers-b" data-show="optB"></div>', '<div id="numbers-b" data-show="optB">' + numbers_b + '</div>'),
                 ('<div id="numbers-c" data-show="florc"></div>', '<div id="numbers-c" data-show="florc">' + numbers_c + '</div>'),
                 ('<div id="numbers-c8" data-show="florc8"></div>', '<div id="numbers-c8" data-show="florc8">' + numbers_c8 + '</div>'),
                 ('<table id="mat-table"></table>', '<table id="mat-table">' + tbl + '</table>'),
                 ('<table id="mat-table-b"></table>', '<table id="mat-table-b">' + tbl_b + '</table>'),
                 ('<ul id="scope"></ul>', '<ul id="scope">' + scope + '</ul>'),
                 ('<script>/*PANEL*/</script>\n', '')):
    assert old in html, old
    html = html.replace(old, new)
html = html.replace('/*DATA*/', json.dumps(data, separators=(',', ':')).replace('</', '<\\/'))
html = html.replace('/*BUNDLE*/', (W / 'viewer/bundle.js').read_text().replace('</script', '<\\/script'))
html += '\n<!-- three.js r160 · MIT License · Copyright 2010-2023 three.js authors. Montserrat · SIL Open Font License 1.1 -->\n'
out = W / 'viewer/Abrir-Flor-Central-F09-Opciones-A-B-C.html'
out.write_text(html)
print('escrito', out, round(out.stat().st_size / 1e6, 2), 'MB')
