# Bitàcola d'experiments — Grup 07 (QuickDraw)

Document viu on anem registrant tots els experiments. Una entrada per run
significativa. Les runs trivials (smoke tests, debugging) no cal posar-les.

> **Convenció**: per cada experiment, copiem la URL de W&B i una captura
> dels gràfics rellevants a `assets/` (carpeta ignorada per git si és gran).

---

## Com afegir un experiment

1. Pensa la **hipòtesi** abans de llançar-lo: què esperes que passi i per què?
2. Llança la run amb un `--run_name` descriptiu i `--tags`.
3. Quan acabi, copia la **URL de W&B** i pega-la sota.
4. Fes captures dels panels més importants (`val/acc`, `train/loss`, etc.).
5. Escriu **3 línies** d'observacions: què ha funcionat, què no, i el següent pas.

Plantilla buida al final del document — copia-la i omple-la.

---

## Sumari ràpid

| ID  | Data       | Run                  | Classes | Img | Epochs | val_acc | top3    | Notes                                       |
|-----|------------|----------------------|---------|-----|--------|---------|---------|---------------------------------------------|
| E01 | 2026-04-27 | local-smoke-test     | 3       | 32  | 2      | 84.00%  | 100%    | CPU baseline, valida pipeline + W&B         |
| E02 | 2026-04-27 | local-smoke-mps      | 3       | 32  | 2      | 83.33%  | 100%    | MPS activat, no speedup amb dades petites   |
| E03 | 2026-04-27 | local-10cls-64px     | 10      | 64  | 5      | 91.73%  | 98.87%  | Primer baseline real local, MPS aporta clar |

---

## E01 — Smoke test inicial (CPU)

- **Run W&B**: <https://wandb.ai/glogc-uab/quickdraw-xn-g07/runs/ohgvh9ax>
- **Data**: 2026-04-27
- **Hipòtesi**: el `train.py` integrat amb W&B funciona end-to-end amb 3
  classes i pipeline complet (load → train → eval → log → checkpoint).
- **Setup**:
  ```
  --max_items_per_class 500 --image_size 32 --epochs 2 --batch_size 64
  Classes: apple, cat, dog
  Device: cpu
  ```
- **Resultats**:
  - `val_acc` final: **84%** (de 44% → 84% en 2 epochs)
  - `train_acc` final: 78%
  - `top3_acc`: 100% (esperable amb 3 classes)
  - Temps per època: ~29s
- **Observacions**:
  - Pipeline OK, W&B sincronitza, gradients i mètriques de sistema també.
  - El gap train→val és petit, no hi ha overfit.
  - Valida que podem passar al següent pas.
- **Següent**: provar MPS per veure si guanyem velocitat.

---

## E02 — Smoke test amb MPS

- **Run W&B**: <https://wandb.ai/glogc-uab/quickdraw-xn-g07/runs/kvzinj8x>
- **Data**: 2026-04-27
- **Hipòtesi**: utilitzant la GPU del Mac (MPS), les èpoques han de
  baixar significativament respecte a CPU.
- **Setup**: idèntic a E01 però amb `train.py` actualitzat per
  detectar MPS.
  ```
  Device: mps
  ```
- **Resultats**:
  - `val_acc` final: **83.33%** (variància normal vs E01)
  - Temps per època 1: 34.4s (warmup MPS)
  - Temps per època 2: 27.6s
- **Observacions**:
  - MPS s'activa correctament. NO hi ha speedup notable.
  - Coll d'ampolla: el `_draw_strokes_to_image` es fa al `DataLoader`
    workers (CPU) i el model és tan petit que MPS no aporta.
  - **Hipòtesi per al següent**: amb `image_size=64` i model més gran,
    MPS hauria de guanyar.
- **Següent**: pujar a 10 classes, `image_size=64`, 8 epochs.

---

## E03 — Primer baseline real (10 classes, 64×64)

- **Run W&B**: <https://wandb.ai/glogc-uab/quickdraw-xn-g07/runs/e2b64esp>
- **Data**: 2026-04-27
- **Hipòtesi**: amb 10 classes, imatges 64×64 i 1500 mostres per classe
  (15k total), MPS hauria d'aportar speedup respecte a CPU i el model
  hauria d'arribar a `val_acc` > 80%.
- **Setup**:
  ```
  --max_items_per_class 1500 --image_size 64 --epochs 5 --batch_size 128
  --num_workers 4
  Classes: les 10 primeres de categories.txt
  Device: mps
  ```
- **Resultats**:
  - `val_acc` per època: 81.9% → 85.9% → 89.9% → 90.9% → **91.7%**
  - `train_acc` final: 93.2%
  - `top3_acc` final: **98.9%**
  - `epoch_time`: 62s (warmup) → 58s estabilitzat
  - Best checkpoint desat: `checkpoints/best_e2b64esp.pth`
- **Observacions**:
  - MPS sí que aporta amb dades realistes — clarament millor que CPU
    (per a aquest mateix volum, CPU passaria de 4-5 min/època).
  - Gap train→val petit (93.2% vs 91.7%) → **no hi ha overfit encara**;
    podria entrenar més epochs i seguir guanyant accuracy.
  - Loss baixant monòtonament a `train` i `val` → l'optimitzador i el
    learning rate van bé.
  - El `train/lr` ha baixat de 1e-3 a 2.5e-4 durant les 5 epochs
    (StepLR amb step=2, gamma=0.5).
- **Següent**:
  - Pujar a **50 o 100 classes** per veure com cau l'accuracy.
  - Provar 8-10 epochs amb les mateixes 10 classes per saber el "sostre"
    abans d'overfit.
  - Afegir `RandomAffine` al transform per generalitzar millor.

---

## Plantilla per als pròxims experiments

Copia això sota i omple-ho:

```markdown
## E0X — <títol curt>

- **Run W&B**: <URL>
- **Data**: YYYY-MM-DD
- **Hipòtesi**: què esperem que passi i per què.
- **Setup**:
  ```
  Comand exacte o paràmetres clau
  ```
- **Resultats**:
  - val_acc / top3 / loss
  - temps per època
- **Observacions**:
  - 3 línies de què hem après.
- **Següent**: què provem ara.
```

---

## Idees pendents (backlog)

- [ ] Pujar a 10 classes en local i validar speedup MPS amb `image_size=64`.
- [ ] Extreure tot el dataset (350 cls) — primer en local per validar, després a la VM.
- [ ] Llançar primer sweep complet (`sweep.yaml`) un cop tinguem la VM.
- [ ] Provar **data augmentation** (`RandomAffine`, blur lleuger) per generalitzar als dibuixos amb el dit.
- [ ] Comparar **CNN bitmap vs RNN/LSTM sobre strokes** (idea final del projecte).
- [ ] Anti-aliasing al render: `Image.new` actual usa línies serrades; provar `cv2.line(..., LINE_AA)`.
- [ ] Curriculum learning: començar amb classes "fàcils" (objectes simètrics) i pujar.
- [ ] Mesurar inference time per saber si pot anar en temps real amb el dit.

---

## Resultats acumulats

A mesura que tinguem més runs, omplim aquí una taula per veure
l'evolució de la millor `val_acc` segons configuració.

| Setup                         | Classes | val_acc | Comand |
|-------------------------------|---------|---------|--------|
| baseline 3cls 32px 2ep        | 3       | 84.00%  | E01    |
| baseline 3cls 32px 2ep MPS    | 3       | 83.33%  | E02    |
| baseline 10cls 64px 5ep MPS   | 10      | 91.73%  | E03    |
