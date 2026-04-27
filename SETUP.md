# Projecte Deep Learning — Grup 07 (Hand Drawing Tracking)

Guia ràpida d'engegada per a la VM del DCC + W&B.

## 0. Repo

```bash
git clone git@github.com:ED-2526/projecte-deep-learning-07.git
cd projecte-deep-learning-07
```

> **Push amb SSH key**: si necessites una clau nova a la VM,
> `ssh-keygen -t ed25519 -C "gloria@openteca.es"` → copia el contingut
> de `~/.ssh/id_ed25519.pub` a *GitHub → Settings → SSH and GPG keys*.
> Canvia el remote a SSH si encara és HTTPS:
> `git remote set-url origin git@github.com:ED-2526/projecte-deep-learning-07.git`.

## 1. Connexió a la VM (grup 07 → VM_IA_07)

```bash
ssh -J jumper@bgjmpvgpu.uab.cat edxnG07@10.12.7.7
# 1a contrasenya (jumper):  nomar2025
# 2a contrasenya (G07):     ea4aequagh
```

Comprova càrrega de GPUs abans: <http://neptu.uab.es/GPU/dashboard.html>.

## 2. Entorn conda (només la primera vegada per usuari a la VM)

```bash
bash ../datasets/Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc

conda env create -f environment.yml
conda activate xn-g07
```

## 3. Login a W&B (un sol cop per VM/usuari)

L'API key és secreta — no la committegis. Hi ha dues opcions:

```bash
# Opció A: interactiu
wandb login --relogin
# enganxa la key i Enter

# Opció B: via variable d'entorn (útil dins scripts)
export WANDB_API_KEY=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
echo 'export WANDB_API_KEY=...' >> ~/.bashrc
```

Comprova: `wandb status`.

## 4. Dataset

El `.rar` de 22GB no va al repo (ja està a `.gitignore`). Opcions:

1. **Compartit a la VM** (recomanat): un cop us deixin escriure a
   `/home/datasets`, descomprimiu-lo allà i feu servir
   `--dataset_dir=/home/datasets/quickdraw_simplified`.
2. **Personal**: descomprimiu-lo a `~/data/simplified_dataset` i passeu
   `--dataset_dir=~/data/simplified_dataset`.

Necessitareu també `categories.txt` amb una classe per línia (la podeu
generar a partir dels noms `full_simplified_<classe>.ndjson`).

```bash
ls /home/datasets/quickdraw_simplified \
  | sed -E 's/^full_simplified_(.*)\.ndjson$/\1/' > categories.txt
```

## 5. Una run "normal"

```bash
python train.py \
    --epochs 8 --batch_size 128 --lr 1e-3 \
    --max_items_per_class 2500 --image_size 64 \
    --run_name "baseline-cnn-350cat" \
    --tags baseline cnn
```

S'envia tot a W&B: loss, accuracy, top-3, lr, gradients, temps per època
i el millor checkpoint com a *artifact* (afegeix `--log_artifact`).

## 6. Experiments amb sweeps (recomanat)

```bash
# 1) Crea el sweep (un sol cop)
wandb sweep sweep.yaml          # imprimeix un SWEEP_ID

# 2) Llança agents (un per VM, o varis a la mateixa VM si la GPU aguanta)
wandb agent <ENTITY>/quickdraw-xn-g07/<SWEEP_ID>
```

Coses que el sweep ja explora:

- `lr` (log-uniform 1e-4 … 5e-3)
- `batch_size` 64 / 128 / 256
- `dropout` 0.2–0.5
- `fc_size` 256 / 512 / 1024
- `optimizer` adam / sgd
- `weight_decay` 0 / 1e-5 / 1e-4
- `image_size` 64 / 96
- `line_width` 3 / 5 / 7 (afecta com de gruixuts són els traços renderitzats)
- `lr_gamma` 0.3 / 0.5 / 0.7

Hyperband talla aviat les runs dolentes.

## 7. Idees concretes per millorar el model

Mirant els resultats de l'Enric (60% top-1 amb 350 categories), els
problemes clars i què intentaríem:

1. **Class overlap** (poma↔ceba, pizza↔pilota): augmentar capacitat
   (afegir un 4t bloc conv 128→256, o `fc_size=1024`) i pujar
   `image_size` a 96 → més detall.
2. **Línies massa fines a 64×64**: variar `line_width` al sweep i provar
   *anti-aliasing* canviant `Image.new("L", ...)` per render amb `cairo`
   o `cv2.line` amb `LINE_AA`.
3. **Data augmentation**: `RandomAffine(degrees=10, translate=(0.05,0.05),
   scale=(0.9,1.1))` abans del `ToTensor`. Important per generalitzar
   als dibuixos fets amb el dit.
4. **Per a la idea final** (endevinar mentre dibuixes): canviar a
   **RNN/LSTM o Transformer** sobre la seqüència de strokes en lloc de
   bitmap. El dataset ja porta els strokes com a llistes ordenades — és
   gratis i és el que fa l'API original de QuickDraw.
5. **MediaPipe** per al *finger tracking*: separa-ho en mòdul a part
   (`tracker.py`) per no barrejar-ho amb l'entrenament.

## 8. Estructura recomanada del repo

```
projecte-deep-learning-07/
├── train.py             ← entrenament + W&B
├── sweep.yaml           ← config de sweeps
├── environment.yml
├── .gitignore
├── README.md            ← portada del projecte (substituir l'exemple)
├── SETUP.md             ← aquesta guia
├── models/
│   └── cnn.py           ← (opcional) treure la classe del train.py
├── data/
│   └── dataset.py       ← (opcional) treure la classe del train.py
├── tracker/
│   └── finger_tracker.py← MediaPipe → strokes
├── notebooks/
│   └── exploracio.ipynb
└── checkpoints/         ← ignorat per git
```

## 9. Quan acabeu de treballar

`exit` la sessió SSH **i** atureu la VM a <https://labs.azure.com/virtualmachines>
(sinó es continuen comptant les hores).
