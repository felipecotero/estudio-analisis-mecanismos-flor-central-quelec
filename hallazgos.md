# Flor central · F09 · Opciones A y B con física (11 nov 2026)

Simulación física de un pétalo: tela de 80 g/m², aro de 2 m, vuelo hasta 4 × 3,5 m, bisagra en el aro periférico, elástico y rienda. El visor con las dos opciones es `index.html`. La primera versión, solo con la opción A, se conserva como `version-anterior-solo-opcion-A.html`. El código está en `codigo/`.

## Opción A · Aro de 2 m + vuelo (Plan A de la reunión del 5 nov)

1. **El aro solo no se sostiene en voladizo.** Sujeto por la raíz y llevado a 12° sobre la horizontal, el borde lejano cae 1,8–1,9 m: queda colgando. Pasa con fleje 1070 de 15 × 0,8 y 15 × 1,0 mm y con varilla de fibra de vidrio de 6 y 8 mm. Causa: junto a la raíz el aro corre paralelo a la bisagra y sostener el pétalo le exige torsión, que ninguno tiene. Un aro rígido de tubo de aluminio de 19 × 1,5 mm cede 0,5 m. El fleje de 15 × 0,5 mm quedó sin dato (el cálculo no convergió).
2. **Con cruceta sí (A2).** Brazo de tubo de aluminio de 25 mm de raíz a borde lejano (2 m) más travesaño de varilla de fibra de vidrio de 8 mm. Los lados del aro caen 5 cm y la tela del disco hace una bolsa de 5 cm. El aro pasa a dar solo el contorno; fleje o varilla dan casi lo mismo.
3. **El vuelo de tela sin apoyo cuelga** 1,4 m bajo la raíz, como una falda. Desde abajo se leen cinco discos de 2 m.
4. **Con antena de punta (A3)** el pétalo llega a 3,9 m de largo por unos 2,3 m de ancho. La punta cae 0,64 m con varilla de 8 mm y 0,35 m con 10 mm. Su plegado no está simulado en esta opción.

### Mecanismo A2 (fleje 15 × 0,8)

- Cinco bisagras en el aro periférico, escalonadas 6 cm en altura. Los aros se voltean hacia el centro y se apilan planos.
- Sector de 30 cm de diámetro solidario a cada raíz; la rienda se arrolla en él y baja por el tubo central.
- Elástico compensador entre una palanca de la raíz y un anclaje bajo la bisagra: levanta el pétalo desde la pila y lo frena cuando cae hacia fuera.
- Tope a 12° y pestillo (por diseñar).

| Dato A2 | Valor |
|---|---|
| Masa por pétalo | 2,4 kg |
| Par del peso al levantar / ya abierto | 22 / 26 N·m |
| Par máximo del elástico | 27 N·m |
| Tensión máxima de una rienda | 70 N |
| Cinco riendas a un mando, escalonadas 4 s | 141 N |
| Recorrido de rienda por pétalo | 44 cm |
| Paquete guardado | 2,31 m de diámetro × 0,53 m de alto sobre el piso de la plataforma |
| Sujeción con el pétalo abierto | 12 N·m: necesita pestillo |

## Opción B · Flex en todo el contorno

Base A2 más una antena de punta y un flex por todo el contorno, ambos de varilla de fibra de vidrio de 6 mm. El pétalo queda completo y tenso.

- **Flex de contorno:** una sola varilla de 11,5 m en una funda por todo el borde, un 1 % más larga que el borde. Punta redondeada con arco de 50 cm de radio (la varilla no puede hacer un pico).
- **Antena de punta:** varilla de 2 m que prolonga el brazo. Sin ella la punta cuelga casi 2 m: el flex solo no la sostiene de forma fiable.
- **Cordón de contención:** tres ramales por el envés. Los de los lados van al cruce del brazo y el travesaño; el de la punta va a un costado del aro para que la antena se curve de lado.
- **Secuencia:** el pétalo voltea contenido hasta unos 100°, ahí se suelta el cordón y el flex abre hacia arriba; después baja abierto hasta el tope a 12°.
- **Material del flex:** con varilla de fibra de vidrio de 6 u 8 mm la tela se tensa. Con fleje de 15 × 0,8 mm de canto no (la punta cuelga 1,4 m); con fleje de 15 × 2 mm pesa demasiado (punta y lados caen 0,8 m).
- **Dos posturas estables:** armado en plano queda con la punta 0,22 m abajo y los lados 0,46 m abajo; tras desplegar queda con la punta 0,92 m abajo y los lados a nivel.

| Dato B | Valor |
|---|---|
| Masa por pétalo | 3,2 kg |
| Tela por pétalo | 9,8 m² |
| Par del peso al levantar contenido / abierto y desplegado | 47 / 42 N·m |
| Par máximo del elástico | 58 N·m |
| Tensión máxima de una rienda | 160 N |
| Cinco riendas a un mando, escalonadas 4 s | 578 N |
| Cordón de contención, por ramal | 19 N |
| Deformación máxima de la varilla | 1,0 % (en el límite de la fibra de vidrio) |
| Ancho visible / flor de borde a borde | 3,43 m / 9,7 m |
| Paquete guardado | 3,84 m de diámetro; cada pétalo contenido levanta 0,48 m sin aplastar |

## Límites de ambas

- Sin viento. Entre 45 y 49 m² de tela a 10 m; la Compact 12 admite 20 daN de empuje lateral manual y 45 km/h en exterior. Por validar con el proveedor de la máquina.
- Un pétalo simulado y copiado cinco veces: no hay choque entre pétalos ni entre capas. En B los pétalos abiertos se montan unos 25 cm.
- En B la varilla puede cruzarse consigo misma al recogerse; el alto real de la pila no está calculado.
- Brazo de aluminio tomado como rígido.
- Supuestos por medir: bastidor de 2 m de diámetro, piso guardado a 1,40 m, tarima a 2,75 m, boca de 2,6 m (A) o 4,1 m (B).

## Pendiente de decidir

- Opción A o B. La B da el pétalo completo pero exige una boca de unos 4 m.
- Cuánto mide de verdad el espacio para guardar la flor: de eso depende que quepa cualquiera de las dos.
