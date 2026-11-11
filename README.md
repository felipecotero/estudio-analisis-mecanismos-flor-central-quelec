# Estudio y análisis de mecanismos de la flor central

**Quelec y la semilla del futuro** · Departamento de arte · 11 de noviembre de 2026

Estudio físico de cómo se sostienen, se guardan y se abren los cinco pétalos de la flor central (4 m de largo × 3,5 m de ancho cada uno) sobre una plataforma de tijera Haulotte Compact 12.

Cada pétalo sale de una simulación: tela, aro, varillas, peso, bisagra, elástico y cuerdas. Nada está animado a mano.

## Cómo verlo

Abre `index.html` en un navegador con WebGL. Funciona sin Internet. Trae las dos opciones en la misma página.

`version-anterior-solo-opcion-A.html` es la primera versión del visor, solo con la opción A.

## Opción A · Aro de 2 m + vuelo de tela

Es el Plan A de la reunión del 5 de noviembre.

- **A1 · Aro + vuelo.** El aro no se sostiene sujeto solo por la raíz: queda colgando. Pasa con fleje 1070 de 0,8 y 1,0 mm y con varilla de fibra de vidrio de 6 y 8 mm.
- **A2 · Aro + cruceta.** Un brazo de tubo de aluminio y un travesaño de varilla cargan el peso. Se sostiene, abre y se recoge con la misma rienda. El vuelo de tela sin apoyo cuelga como una falda.
- **A3 · Cruceta + antena de punta.** Una varilla lleva la tela hasta la punta. Solo postura abierta.

## Opción B · Flex en todo el contorno

La A2 más una antena de punta y un flex por todo el borde, ambos de varilla de fibra de vidrio de 6 mm. Un cordón recoge el contorno en tres lazos; el pétalo voltea contenido, casi vertical se suelta el cordón, el flex abre la tela hacia arriba y el pétalo baja abierto. Queda completo y tenso.

## Comparación

| | A2 | B |
|---|---|---|
| Masa por pétalo | 2,4 kg | 3,2 kg |
| Paquete guardado, diámetro | 2,31 m | 3,84 m |
| Rienda, un pétalo | 70 N | 160 N |
| Rienda, las cinco a un mando | 141 N | 578 N |
| Par máximo del elástico | 27 N·m | 58 N·m |
| Flor abierta, de borde a borde | 7,0 m | 9,7 m |

El detalle está en [`hallazgos.md`](hallazgos.md).

## Qué no cubre

- **Viento.** Es el riesgo mayor: entre 45 y 49 m² de tela a 10 m de altura. Hay que resolverlo con el proveedor de la máquina.
- **Choques** entre pétalos vecinos y entre capas de la pila. Se simula un pétalo y se copia cinco veces.
- **En la opción B,** la varilla trabaja en su límite de deformación (1 %) y el alto real de la pila no está calculado.
- **Tallo, hojas de bore y compuerta de la trampa.**
- Las medidas de bastidor, piso guardado, tarima y boca son supuestas hasta que se midan la máquina y la trampa.

Esto es un estudio para orientar el prototipo. No es un cálculo estructural ni una certificación.

## Contenido

- `index.html`: visor con las opciones A y B.
- `capturas/`: nueve vistas del visor.
- `hallazgos.md`: resultados y números.
- `codigo/simulacion/`: motor físico en Python (JAX, NumPy, SciPy).
  - `mesh.py`: patrón del pétalo.
  - `petal.py`: física y materiales.
  - `configs.py`: variantes.
  - `run_anim.py`, `run_poses.py`: opción A.
  - `run_anim_flex.py`, `rerun_flex_open.py`, `run_poses_flex.py`: opción B.
- `codigo/visor/`: fuente del visor (three.js) y `build.py`, que lo empaqueta. Las rutas de `build.py` apuntan al entorno donde se generó y hay que ajustarlas para correrlo en otro equipo.

## Créditos

Mecanismo: Jahir Santos. Dirección de arte: Mario Ávila. Asistencia de dirección de arte y coordinación: Felipe Camacho Otero. Simulación y visor hechos con Claude.

El visor incluye three.js r160 (licencia MIT) y la tipografía Montserrat (SIL Open Font License 1.1).
