# 🔬 Classification de Cancers : Framework Deep Learning Multi-Architectures

Ce projet est un pipeline MLOps de bout en bout dédié à la classification d'images histopathologiques (poumon et côlon). Il permet d'entraîner et de comparer de manière rigoureuse des architectures classiques (CNN) avec les dernières avancées du Deep Learning (Vision Transformers, State Space Models, Kolmogorov-Arnold Networks).

---

## 🏗️ Architecture du Projet

Le framework est segmenté de manière modulaire pour séparer la donnée, la logique métier et l'analyse.

```text
📁 projet-cancer/
│
├── 📁 configs/                 # 🎛️ Configuration centrale
│   └── config.py               # Le "cerveau" : hyperparamètres et réglages du pipeline
│
├── 📁 src/                     # 🧠 Code Source
│   ├── 📁 data/                # Pipeline de données (Download, Split, Preprocess, Loader)
│   ├── 📁 models/              # Zoo de modèles (Baseline, Transformers, Hybrids, SSM, KAN)
│   ├── 📁 engine/              # Moteurs d'exécution (Train, Evaluate, Grid Search)
│   └── 📁 utils/               # Outillage (Metrics, Visualizer, Complexity, FLOPs)
│
├── 📁 data/                    # 🗂️ Données (généré automatiquement)
│   ├── raw/                    # Données brutes (Kaggle)
│   ├── split/                  # Train / Validation / Test
│   └── preprocessed/           # Images transformées (Resize, CLAHE, etc.)
│
├── 📁 results/                 # 📊 Sorties (généré automatiquement)
│   ├── 📁 checkpoints/         # Modèles sauvegardés (.pth)
│   └── 📁 metrics/             # CSV, matrices de confusion, graphiques
│       └── complexity/         # Paramètres et FLOPs
│
├── main.py                     # 🚀 Point d'entrée unique
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation
```

---

## 🛠️ Installation et Prérequis

### 1. Prérequis Système

* Python 3.8+
* Environnement CUDA (fortement recommandé pour accélérer l'entraînement)
* API Kaggle configurée :

  * Placez `kaggle.json` dans :

    * `~/.kaggle/` (Linux/Mac)
    * `%USERPROFILE%\.kaggle\` (Windows)

---

### 2. Installation

```bash
# Installation des dépendances standards
pip install -r requirements.txt

# Installation spécifique pour l'architecture KAN
pip install git+https://github.com/Blealtan/efficient-kan.git
```

---

## 🎛️ Configuration et Paramétrage (`config.py`)

Le fichier `configs/config.py` centralise toute la logique du projet.

---

### 🚀 Pilotage de l'Exécution

* `MODE` : Définit l'action du pipeline

  * `download` / `preprocess` : préparation des données
  * `train` : entraînement simple
  * `grid_search` : recherche d'hyperparamètres
  * `eval` : évaluation finale
  * `compare` : analyse comparative
  * `all` : pipeline complet automatique (download ➔ preprocess ➔ train ➔ eval)

* `APPROACH` : famille de modèles (`baseline`, `transformers`, `hybrids`, `ssm`, `kan`)

* `MODEL_NAME` : architecture spécifique (`resnet18`, `vit_b_16`, `vision_mamba`, `vision_kan`)

---

### 📊 Définition de la Tâche (Données)

* `TASK` : `"colon"`, `"lung"` ou `"all"`
* `IMG_SIZE` : résolution des images (ex: `(224, 224)`)
* `BATCH_SIZE` : taille des batchs (dépend de la VRAM)

---

### 🧠 Hyperparamètres d'Entraînement

* `EPOCHS`
* `LEARNING_RATE`
* `OPTIMIZER` (`adam`, `adamw`, `sgd`)
* `WEIGHT_DECAY`

---

### 🛡️ Optimisation et Régularisation

* `USE_LR_SCHEDULER` : ajuste automatiquement le learning rate
* `USE_EARLY_STOPPING` : stoppe si stagnation
* `EARLY_STOPPING_PATIENCE` : nombre d’epochs tolérées
* `USE_GRAD_CLIPPING` : stabilise les gradients (important pour Transformers / Mamba)

---

### 🔬 Recherche d'Hyperparamètres

* `GRID_SEARCH_PARAMS` : liste de dictionnaires (combinaisons testées automatiquement)

---

## 🚀 Guide de Running

Lancez le pipeline via :

```bash
python main.py
```

---

### 📋 Workflow recommandé

1. **Préparation des données**

   * `MODE = "download"`
   * `MODE = "preprocess"`

2. **Recherche d’hyperparamètres**

   * `MODE = "grid_search"`

3. **Entraînement final**

   * `MODE = "train"`

4. **Évaluation**

   * `MODE = "eval"`

5. **Comparaison globale**

   * `MODE = "compare"`

---

## ⚖️ Évaluation de l’Efficience

Pour comparer équitablement les architectures, le projet mesure automatiquement :

* **Nombre de paramètres (Millions)** → taille mémoire
* **FLOPs** → coût computationnel

Résultats disponibles dans :

```
results/metrics/complexity/
```

---

## 📊 Sorties Générées

* Courbes d’apprentissage (train/validation)
* Matrices de confusion
* Scores : Accuracy, Precision, Recall, F1, AUC
* Comparaisons inter-modèles
* Comparaisons intra-modèles (grid search)

---

