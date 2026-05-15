# Phishing Detection — End-to-End Colab Notebook
### "A Lightweight Hybrid Machine Learning Framework for Phishing URL Detection and Classification"
**Runtime:** T4 GPU | **Author:** BE CSE, Chitkara University

---

## CELL 0 — 🔁 Drive Mount, Checkpoint Load & Smart Restoration

```python
# ============================================================
# CELL 0 — Drive Mount, Checkpoint Load & Smart Restoration
# Run this cell FIRST on every session revisit.
# ============================================================

# ── Step 1: Mount Google Drive ───────────────────────────────
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("✅ Drive mounted at /content/drive")

# ── Step 2: Global Path Constants ───────────────────────────
import os, json, shutil, joblib, pickle, warnings, time
from datetime import datetime

DRIVE_BASE       = "/content/drive/MyDrive/PhishingDetection"
CHECKPOINT_PATH  = f"{DRIVE_BASE}/checkpoints/checkpoint_state.json"
DATASETS_DIR     = f"{DRIVE_BASE}/datasets"
CHECKPOINTS_DIR  = f"{DRIVE_BASE}/checkpoints"
MODELS_DIR       = f"{DRIVE_BASE}/models"
METRICS_DIR      = f"{DRIVE_BASE}/metrics"
PLOTS_DIR        = f"{DRIVE_BASE}/plots"
RESULTS_DIR      = f"{DRIVE_BASE}/results"
RANDOM_SEED      = 42
TARGET_COL_1     = None   # restored below if already detected
TARGET_COL_2     = None   # restored below if already detected

# ── Step 3: Create Folder Structure ─────────────────────────
for folder in [DATASETS_DIR + "/dataset1",
               DATASETS_DIR + "/dataset2",
               CHECKPOINTS_DIR,
               MODELS_DIR,
               METRICS_DIR,
               PLOTS_DIR,
               RESULTS_DIR]:
    os.makedirs(folder, exist_ok=True)
print("📁 Drive folder structure verified")

# ── Step 4: Helper Functions ─────────────────────────────────
DEFAULT_STATE = {
    "drive_mounted": False,
    "dependencies_installed": False,
    "datasets_downloaded": False,
    "dataset1_loaded": False,
    "dataset2_loaded": False,
    "eda_done": False,
    "eda_ds2_done": False,
    "preprocessing_done": False,
    "preprocessing_ds2_done": False,
    "smote_done": False,
    "lr_trained": False,
    "rf_trained": False,
    "xgb_trained": False,
    "hybrid_trained": False,
    "cv_done": False,
    "results_consolidated": False,
    "plots_done": False,
    "shap_done": False,
    "cross_dataset_done": False,
    "inference_demo_done": False,
    "final_save_done": False,
    "metadata": {
        "TARGET_COL_1": None,
        "TARGET_COL_2": None,
        "dataset1_shape": None,
        "dataset2_shape": None
    },
    "last_updated": None,
    "session_notes": ""
}

def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_STATE.copy()

def save_checkpoint(state: dict):
    state["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(state, f, indent=2)

def mark_done(key: str):
    state = load_checkpoint()
    state[key] = True
    save_checkpoint(state)

def is_done(key: str) -> bool:
    state = load_checkpoint()
    return state.get(key, False)

def drive_path(relative: str) -> str:
    return f"{DRIVE_BASE}/{relative}"

def save_to_drive(obj, relative_path: str):
    full_path = drive_path(relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if full_path.endswith(".json"):
        with open(full_path, "w") as f:
            json.dump(obj, f, indent=2)
    else:
        print(f"💾 Saving to Drive: {full_path} ...")
        joblib.dump(obj, full_path)
        print("✅ Saved.")

def load_from_drive(relative_path: str):
    full_path = drive_path(relative_path)
    if not os.path.exists(full_path):
        return None
    if full_path.endswith(".json"):
        with open(full_path, "r") as f:
            return json.load(f)
    return joblib.load(full_path)

# ── Step 8: Silent dependency install ───────────────────────
import subprocess, sys, importlib.util

def _silent_install(pkg):
    # Use find_spec — does NOT execute the package's __init__.py
    module_name = pkg.replace("-", "_").split("[")[0]
    if importlib.util.find_spec(module_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["xgboost", "shap", "imbalanced-learn", "optuna", "kaggle", "kaleido"]:
    _silent_install(pkg)

# ── Step 7: All Imports ──────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-whitegrid')

# ── Step 5: Load Checkpoint & Print Progress Report ─────────
state = load_checkpoint()

STEP_LABELS = {
    "drive_mounted":         "Drive Mounted",
    "dependencies_installed":"Dependencies Installed",
    "datasets_downloaded":   "Datasets Downloaded",
    "dataset1_loaded":       "Dataset 1 Loaded & Inspected",
    "dataset2_loaded":       "Dataset 2 Loaded & Inspected",
    "eda_done":              "EDA Complete (DS1)",
    "eda_ds2_done":          "EDA Complete (DS2)",
    "preprocessing_done":    "Preprocessing Done (DS1)",
    "preprocessing_ds2_done":"Preprocessing Done (DS2)",
    "smote_done":            "SMOTE Applied",
    "lr_trained":            "Logistic Regression Trained",
    "rf_trained":            "Random Forest Trained",
    "xgb_trained":           "XGBoost Trained",
    "hybrid_trained":        "Hybrid Voting Classifier Trained",
    "cv_done":               "Cross-Validation Done",
    "results_consolidated":  "Results Table Built",
    "plots_done":            "Plots Done",
    "shap_done":             "SHAP Explainability Done",
    "cross_dataset_done":    "Cross-Dataset Validation Done",
    "inference_demo_done":   "Inference Demo Done",
    "final_save_done":       "Final Save Done",
}

CELL_MAP = {
    "drive_mounted":         "Cell 1",
    "dependencies_installed":"Cell 2",
    "datasets_downloaded":   "Cell 4",
    "dataset1_loaded":       "Cell 5",
    "dataset2_loaded":       "Cell 6",
    "eda_done":              "Cell 7",
    "eda_ds2_done":          "Cell 8",
    "preprocessing_done":    "Cell 9",
    "preprocessing_ds2_done":"Cell 10",
    "smote_done":            "Cell 11",
    "lr_trained":            "Cell 12",
    "rf_trained":            "Cell 13",
    "xgb_trained":           "Cell 14",
    "hybrid_trained":        "Cell 15",
    "cv_done":               "Cell 16",
    "results_consolidated":  "Cell 17",
    "plots_done":            "Cell 18",
    "shap_done":             "Cell 19",
    "cross_dataset_done":    "Cell 20",
    "inference_demo_done":   "Cell 21",
    "final_save_done":       "Cell 22",
}

print("╔══════════════════════════════════════════════════╗")
print("║      PHISHING DETECTION — SESSION RESUME         ║")
print("╚══════════════════════════════════════════════════╝\n")

next_cell = None
for key, label in STEP_LABELS.items():
    done = state.get(key, False)
    mark = "✅" if done else "❌"
    suffix = "" if done else " (not yet)"
    print(f"  {mark}  {label}{suffix}")
    if not done and next_cell is None:
        next_cell = CELL_MAP.get(key, "?")

last_ts = state.get("last_updated") or "never"
print(f"\n  ▶  CONTINUE FROM: {next_cell or 'All Done!'}")
print(f"  🕒  Last session: {last_ts}\n")

# ── Step 6: Auto-Restore Session Variables ───────────────────
print("🔄 Restoring session variables...")

def _try_load(rel, varname):
    obj = load_from_drive(rel)
    if obj is not None:
        print(f"  ✅  {varname}  ← loaded from Drive")
    else:
        print(f"  ⚠️   {varname} not found on Drive")
    return obj

# Restore metadata (target columns)
if state["metadata"].get("TARGET_COL_1"):
    TARGET_COL_1 = state["metadata"]["TARGET_COL_1"]
if state["metadata"].get("TARGET_COL_2"):
    TARGET_COL_2 = state["metadata"]["TARGET_COL_2"]

# Preprocessed splits DS1
if state.get("preprocessing_done"):
    X1_train = _try_load("checkpoints/preprocessed_X1_train.pkl", "X1_train")
    X1_test  = _try_load("checkpoints/preprocessed_X1_test.pkl",  "X1_test")
    y1_train = _try_load("checkpoints/preprocessed_y1_train.pkl", "y1_train")
    y1_test  = _try_load("checkpoints/preprocessed_y1_test.pkl",  "y1_test")
    scaler1  = _try_load("checkpoints/scaler1.pkl",               "scaler1")

# Preprocessed splits DS2
if state.get("preprocessing_ds2_done"):
    X2_train = _try_load("checkpoints/preprocessed_X2_train.pkl", "X2_train")
    X2_test  = _try_load("checkpoints/preprocessed_X2_test.pkl",  "X2_test")
    y2_train = _try_load("checkpoints/preprocessed_y2_train.pkl", "y2_train")
    y2_test  = _try_load("checkpoints/preprocessed_y2_test.pkl",  "y2_test")
    scaler2  = _try_load("checkpoints/scaler2.pkl",               "scaler2")

# SMOTE splits
if state.get("smote_done"):
    X1_train_sm = _try_load("checkpoints/X1_train_smote.pkl",  "X1_train_sm")
    y1_train_sm = _try_load("checkpoints/y1_train_smote.pkl",  "y1_train_sm")

# Models
if state.get("lr_trained"):
    lr_model   = _try_load("models/lr_model.pkl",     "lr_model")
    lr_metrics = _try_load("metrics/lr_metrics.json", "lr_metrics")
if state.get("rf_trained"):
    rf_model   = _try_load("models/rf_model.pkl",     "rf_model")
    rf_metrics = _try_load("metrics/rf_metrics.json", "rf_metrics")
if state.get("xgb_trained"):
    xgb_model   = _try_load("models/xgb_model.pkl",     "xgb_model")
    xgb_metrics = _try_load("metrics/xgb_metrics.json", "xgb_metrics")
if state.get("hybrid_trained"):
    hybrid_model   = _try_load("models/hybrid_model.pkl",     "hybrid_model")
    hybrid_metrics = _try_load("metrics/hybrid_metrics.json", "hybrid_metrics")
if state.get("cv_done"):
    cv_results = _try_load("metrics/cv_results.json", "cv_results")
if state.get("results_consolidated"):
    _results_raw = _try_load("results/comparison_table.csv", "comparison_table")
    if _results_raw is not None:
        results_df = pd.read_csv(drive_path("results/comparison_table.csv"))

# Mark drive as mounted
mark_done("drive_mounted")

print(f"\n✅ Session restoration complete. Jump to {next_cell or 'all done!'} to continue.")
print("\n🚀 Environment ready. All available checkpoints restored.")
print("   Run Cell 0 only — then jump directly to your next incomplete cell.")
```

> ### 🧠 Cell 0 Explanation
>
> This is the **master setup cell** — the very first thing you run every time you open this notebook. Think of it as the "startup routine" that wakes everything up and tells you exactly where you left off last time.
>
> **What it does, step by step:**
>
> **Connecting to Google Drive** — `drive.mount('/content/drive')` connects your Google Drive to the Colab session, like plugging in a USB drive. This is needed because Colab's memory wipes itself when the session ends, so all important files (datasets, trained models, results) are stored on Drive for safety.
>
> **Setting folder paths** — Variables like `DRIVE_BASE`, `MODELS_DIR`, `PLOTS_DIR` etc. are just shortcut addresses (strings) pointing to specific folders on your Drive. Instead of typing the full folder path every time, you just use the variable name.
> - `RANDOM_SEED = 42` — Sets a fixed "lucky number" so that any randomness in the code (shuffling data, initializing models) produces the same result every run. This makes experiments reproducible.
>
> **Creating folders** — `os.makedirs(folder, exist_ok=True)` creates the required folder structure on your Drive if it doesn't already exist. `exist_ok=True` means it won't crash if the folder is already there.
>
> **Checkpoint system (the "save game" mechanism)** — `DEFAULT_STATE` is a dictionary (a list of key-value pairs) that tracks whether each step of the project has been completed. Every key like `"lr_trained"` or `"eda_done"` is either `True` (done) or `False` (not done yet). This lets you resume from wherever you stopped without re-running everything.
> - `load_checkpoint()` — Opens and reads the saved progress file (`checkpoint_state.json`) from Drive. `json.load(f)` reads a JSON file (a plain text file structured like a Python dictionary) and returns it as a Python object.
> - `save_checkpoint()` — Writes the current progress dictionary back to Drive as a JSON file. `json.dump(state, f, indent=2)` converts the dictionary to nicely formatted text and saves it.
> - `mark_done(key)` — Marks a specific step as completed in the checkpoint file.
> - `is_done(key)` — Returns `True` or `False` telling you whether a step is already finished.
> - `save_to_drive(obj, path)` — Saves any Python object (like a trained model) to Drive. For `.json` files it uses `json.dump`; for everything else it uses `joblib.dump`, which is a fast way to save large Python objects like machine learning models.
> - `load_from_drive(path)` — Loads a previously saved object back from Drive using `joblib.load` (the reverse of `joblib.dump`).
>
> **Silent dependency installer** — `importlib.util.find_spec(module_name)` checks if a Python package is already installed without actually loading it. If it's missing, `subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])` installs it silently using pip (Python's package manager). Packages installed this way: `xgboost` (gradient boosting model), `shap` (explainability), `imbalanced-learn` (SMOTE for class imbalance), `optuna` (hyperparameter tuning), `kaggle` (to download datasets), `kaleido` (to export Plotly charts as images).
>
> **All imports** — This block loads all the libraries (tools) the entire notebook needs:
> - `numpy (np)` — The fundamental library for numerical math in Python. Used for arrays, matrix operations, and math functions.
> - `pandas (pd)` — Used for working with tabular data (like Excel spreadsheets). Provides the `DataFrame` structure.
> - `scipy.stats` — Statistical functions like skewness, kurtosis, and hypothesis tests.
> - `matplotlib` — The base plotting library. `matplotlib.use('Agg')` sets it to "non-interactive" mode so plots can be saved to files even without a screen display.
> - `matplotlib.pyplot (plt)` — The module within matplotlib used to actually draw charts.
> - `seaborn (sns)` — A higher-level plotting library built on top of matplotlib that makes statistical charts look better with less code.
> - `plotly.express (px)` and `plotly.graph_objects (go)` — Libraries for interactive, web-style charts.
> - `StandardScaler` — Scales all numbers to have mean 0 and standard deviation 1 so models aren't biased by large-valued features.
> - `LabelEncoder` — Converts text labels (like "phishing" / "legitimate") into numbers (0 / 1).
> - `train_test_split` — Splits data into a training set and a testing set.
> - `StratifiedKFold` — Splits data into k folds while maintaining the same class ratio in each fold.
> - `cross_val_score` — Evaluates a model across multiple folds automatically.
> - `LogisticRegression` — A simple but powerful classification model.
> - `RandomForestClassifier` — An ensemble of many decision trees.
> - `VotingClassifier` — Combines multiple models and lets them "vote" on predictions.
> - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report`, `roc_curve` — Various functions to measure how good a model is.
> - `XGBClassifier` — The XGBoost model, one of the best-performing gradient boosting algorithms.
> - `SMOTE` — Synthetic Minority Over-sampling Technique; creates artificial examples of the rare class to fix imbalanced data.
> - `shap` — Explains why a model made a particular prediction (which features mattered most).
> - `warnings.filterwarnings('ignore')` — Suppresses harmless warning messages to keep output clean.
> - `np.random.seed(RANDOM_SEED)` — Seeds numpy's random number generator so random operations are reproducible.
> - `plt.style.use('seaborn-v0_8-whitegrid')` — Sets a clean white grid visual style for all matplotlib charts.
>
> **Progress report** — Loops through all the checkpoint keys and prints a ✅/❌ status for each step, then tells you which cell to run next.
>
> **Auto-restore** — If previous sessions already trained models or preprocessed data, this block automatically loads all those saved objects back into memory so you can continue working with them immediately without retraining.

---

## CELL 1 — Runtime Check & GPU Verification

```python
# ============================================================
# CELL 1 — Runtime Check & GPU Verification
# ============================================================

if is_done("drive_mounted"):
    # Already flagged; just verify GPU anyway on fresh runtime
    pass

import subprocess, psutil, shutil as _shutil

print("=" * 55)
print("  RUNTIME ENVIRONMENT CHECK")
print("=" * 55)

# GPU
gpu_result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"],
                             capture_output=True, text=True)
if gpu_result.returncode == 0:
    print(f"🖥️  GPU     : {gpu_result.stdout.strip()}")
else:
    print("⚠️  No GPU detected — switch runtime to T4 GPU!")

# Python version
import sys
print(f"🐍 Python  : {sys.version.split()[0]}")

# CUDA via torch
try:
    import torch
    print(f"🔥 CUDA    : {'Available ✅' if torch.cuda.is_available() else 'Not available ❌'}")
except ImportError:
    print("🔥 CUDA    : torch not installed (XGBoost will use device='cuda' directly)")

# RAM
ram_gb = psutil.virtual_memory().total / 1e9
print(f"💾 RAM     : {ram_gb:.1f} GB")

# Disk
disk_gb = _shutil.disk_usage('/').free / 1e9
print(f"💿 Disk    : {disk_gb:.1f} GB free")

print("=" * 55)
print("✅ Runtime check complete.\n")
```

> ### 🧠 Cell 1 Explanation
>
> This cell is a **health check** for your computing environment — like a doctor's checkup before starting a demanding project. It confirms that you have the right hardware and software to run this notebook properly.
>
> **`subprocess`** — A built-in Python module that lets Python run terminal/command-line commands, just as if you typed them yourself.
>
> **`psutil`** — A package used for reading system information like how much RAM your machine has. "psutil" stands for "process and system utilities."
>
> **`shutil` (imported as `_shutil`)** — A built-in Python module for file operations. Here it is used specifically to check available disk space with `disk_usage('/')`.
>
> **GPU Check** — `subprocess.run(["nvidia-smi", ...])` runs the `nvidia-smi` terminal command, which is a standard tool provided by NVIDIA to query the GPU's name and total memory. If this command succeeds (`returncode == 0`), a GPU is present. If not, you're warned to switch the Colab runtime to T4 GPU, because training deep models on CPU would be extremely slow.
>
> **Python version** — `sys.version` returns the full Python version string. `.split()[0]` takes just the version number (e.g., "3.10.12").
>
> **CUDA check** — CUDA is NVIDIA's technology that allows software to run computations on the GPU instead of the CPU. `torch.cuda.is_available()` returns `True` if CUDA is accessible via PyTorch. If PyTorch isn't installed, a note is printed saying XGBoost will use CUDA directly via its own `device='cuda'` setting.
>
> **RAM check** — `psutil.virtual_memory().total` returns the total RAM in bytes. Dividing by `1e9` (1 billion) converts it to gigabytes (GB).
>
> **Disk check** — `shutil.disk_usage('/').free` returns the free disk space in bytes at the root directory. Divided by `1e9` gives free GB.
>
> **In plain terms:** This cell answers the question — "Is my machine ready for this job?" It tells you your GPU model, how much RAM you have, how much disk space is free, and what Python version is running. If something critical is missing (like no GPU), you know to fix it before running expensive training cells.

---

## CELL 2 — Install Dependencies

```python
# ============================================================
# CELL 2 — Install Dependencies
# ============================================================

if is_done("dependencies_installed"):
    print("✅ Skipping — dependencies already installed.")
else:
    print("📦 Installing required packages...")
    import subprocess, sys

    packages = [
        "xgboost", "lightgbm", "imbalanced-learn",
        "shap", "optuna", "kaggle", "plotly", "kaleido"
    ]
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    print("✅ All dependencies installed.")
    mark_done("dependencies_installed")
```

> ### 🧠 Cell 2 Explanation
>
> This cell **installs all the extra Python packages** that don't come pre-installed in Google Colab. Think of it like going to the app store and downloading the apps you need before you can use them.
>
> **Checkpoint guard** — `if is_done("dependencies_installed")` checks the saved progress file. If packages were already installed in a previous session, the cell skips the entire installation block to save time.
>
> **`subprocess`** — Again used here to run terminal commands from Python. `subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])` is equivalent to typing `pip install -q xgboost` in the terminal. The `-q` flag means "quiet" — it suppresses all the verbose installation logs.
>
> **`sys.executable`** — Points to the exact Python interpreter currently running, ensuring the package gets installed in the right Python environment.
>
> **Packages being installed:**
> - `xgboost` — A very fast and accurate gradient boosting model, one of the three classifiers used in this project.
> - `lightgbm` — Another gradient boosting model (imported here for potential future use).
> - `imbalanced-learn` — Provides the SMOTE algorithm used later to fix class imbalance.
> - `shap` — Used in Cell 19 to explain what the model is "thinking" when it makes predictions.
> - `optuna` — A framework for automatically finding the best model settings (hyperparameter tuning).
> - `kaggle` — Lets Python talk to the Kaggle platform so we can download datasets programmatically.
> - `plotly` — For creating interactive charts.
> - `kaleido` — A helper library that allows Plotly to save charts as image files (PNG).
>
> **`mark_done("dependencies_installed")`** — After all packages install successfully, this marks the step as complete in the checkpoint file so it won't run again next session.

---

## CELL 3 — Import All Libraries

```python
# ============================================================
# CELL 3 — Import All Libraries
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import time, warnings, os, pickle, joblib, json
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(RANDOM_SEED)
plt.style.use('seaborn-v0_8-whitegrid')

print("✅ All libraries imported successfully.")
```

> ### 🧠 Cell 3 Explanation
>
> This cell **loads (imports) every tool/library** the notebook will use. In Python, `import` is how you bring in pre-written code from other libraries — think of it like loading all your tools onto the workbench before starting a project. Cell 0 already does this too, but this cell ensures a clean standalone import if you're running cells independently.
>
> **`numpy (np)`** — NumPy (Numerical Python). The foundation of scientific computing in Python. It provides the `array` data structure, which is like a super-powered list capable of fast mathematical operations on thousands of numbers at once.
>
> **`pandas (pd)`** — Pandas (Panel Data). Used for working with structured/tabular data. Its main structure, the `DataFrame`, is like an Excel table in Python — rows are data points, columns are features.
>
> **`scipy.stats`** — SciPy's statistics module. Provides advanced statistical functions like computing z-scores, running normality tests, etc.
>
> **`matplotlib`** — The core plotting library in Python. `matplotlib.use('Agg')` switches to a "headless" backend — meaning plots are rendered to image files rather than popping up as windows. This is required in server/cloud environments like Colab when saving files.
>
> **`matplotlib.pyplot (plt)`** — The sub-module of matplotlib used to create and customize charts (bar charts, line plots, histograms, etc.).
>
> **`seaborn (sns)`** — Built on top of matplotlib, seaborn makes it easy to create beautiful statistical plots like heatmaps, box plots, and KDE (distribution) plots with minimal code.
>
> **`plotly.express (px)`** — A high-level library for creating interactive charts (charts you can hover over, zoom in, etc.).
>
> **`plotly.graph_objects (go)`** — The lower-level Plotly interface for more customized interactive charts.
>
> **`StandardScaler`** — Transforms numerical columns so they all have a mean of 0 and standard deviation of 1. This prevents a feature with large values (like URL length = 200) from dominating a feature with small values (like number of dots = 3).
>
> **`LabelEncoder`** — Converts text categories into numbers. E.g., "phishing" → 1, "legitimate" → 0.
>
> **`train_test_split`** — Randomly divides the dataset into two parts: one for training the model and one for testing it. Common split is 80% train, 20% test.
>
> **`StratifiedKFold`** — Divides data into k equal folds for cross-validation, while ensuring each fold has the same proportion of phishing vs. legitimate URLs as the original dataset.
>
> **`cross_val_score`** — Runs a model through k-fold cross-validation and returns the score for each fold.
>
> **`LogisticRegression`** — A classification model that learns a boundary between two classes (phishing vs. legit). Despite the name, it's used for classification, not regression.
>
> **`RandomForestClassifier`** — Trains hundreds of decision trees and combines their votes for the final prediction. More robust and accurate than a single tree.
>
> **`VotingClassifier`** — Combines multiple different models (LR + RF + XGB) and lets them vote. With `voting='soft'`, it averages their confidence scores rather than a simple majority vote.
>
> **`accuracy_score`** — Fraction of predictions that were correct.
> **`precision_score`** — Of all URLs predicted as phishing, how many actually were?
> **`recall_score`** — Of all actual phishing URLs, how many did we catch?
> **`f1_score`** — The harmonic mean of precision and recall — a balanced single score.
> **`roc_auc_score`** — Area Under the ROC Curve; measures the model's ability to distinguish between classes across all thresholds.
> **`confusion_matrix`** — A table showing true positives, false positives, true negatives, and false negatives.
> **`classification_report`** — A text summary of precision, recall, and F1 for each class.
> **`roc_curve`** — Returns data to plot the ROC curve (True Positive Rate vs. False Positive Rate at various thresholds).
>
> **`XGBClassifier`** — Extreme Gradient Boosting classifier. Very fast and accurate, especially on tabular data. Uses GPU (CUDA) when available.
>
> **`lightgbm (lgb)`** — Another gradient boosting library by Microsoft, slightly different from XGBoost but similarly powerful.
>
> **`SMOTE`** — Synthetic Minority Over-sampling Technique. When your dataset has far more legitimate URLs than phishing ones, SMOTE creates synthetic (fake but realistic) phishing examples to balance the classes.
>
> **`shap`** — SHapley Additive exPlanations. Tells you which features pushed the model's prediction toward phishing or legitimate for any given URL. Makes the "black box" model explainable.
>
> **`optuna`** — An automatic hyperparameter optimization framework. `optuna.logging.set_verbosity(optuna.logging.WARNING)` silences its verbose output, showing only warnings.
>
> **`time`** — Built-in module for timing operations (e.g., how long training takes).
> **`warnings`** — Manages Python warning messages. `filterwarnings('ignore')` hides routine warnings.
> **`os`** — Operating system interface; used for file/folder operations.
> **`pickle`** — Serializes Python objects to files (saves/loads objects).
> **`joblib`** — Like pickle but faster for large numpy arrays; preferred for saving ML models.
> **`json`** — Reads and writes JSON files (a human-readable data format).
> **`datetime`** — For working with dates and timestamps.

---

## CELL 4 — Kaggle API Setup & Dataset Download

```python
# ============================================================
# CELL 4 — Kaggle API Setup & Dataset Download
# ============================================================

import os, shutil, zipfile
from google.colab import files

# ── Check if already downloaded and files exist ──────────────
def _datasets_present():
    d1 = f"{DATASETS_DIR}/dataset1"
    d2 = f"{DATASETS_DIR}/dataset2"
    d1_ok = os.path.isdir(d1) and any(f.endswith(".csv") for f in os.listdir(d1))
    d2_ok = os.path.isdir(d2) and any(f.endswith(".csv") for f in os.listdir(d2))
    return d1_ok and d2_ok

if is_done("datasets_downloaded"):
    if _datasets_present():
        print("✅ Datasets already on Drive — skipping download.")
    else:
        print("⚠️  Checkpoint says downloaded but files are missing — re-downloading...")
        import json as _j
        _s = load_checkpoint(); _s["datasets_downloaded"] = False; save_checkpoint(_s)

if not is_done("datasets_downloaded"):
    # ── Upload kaggle.json ───────────────────────────────────
    print("📤 Please upload your kaggle.json file now:")
    uploaded = files.upload()

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_json_dst = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_dst, "wb") as f:
        f.write(uploaded["kaggle.json"])
    os.chmod(kaggle_json_dst, 0o600)
    print("✅ kaggle.json configured.")

    # ── Download datasets ────────────────────────────────────
    tmp_dir = "/content/tmp"
    os.makedirs(f"{tmp_dir}/dataset1", exist_ok=True)
    os.makedirs(f"{tmp_dir}/dataset2", exist_ok=True)

    print("⬇️  Downloading PhiUSIIL dataset...")
    os.system(f"kaggle datasets download -d ndarvind/phiusiil-phishing-url-dataset -p {tmp_dir}/dataset1 --unzip")

    print("⬇️  Downloading Web Page Phishing dataset...")
    os.system(f"kaggle datasets download -d shashwatwork/web-page-phishing-detection-dataset -p {tmp_dir}/dataset2 --unzip")

    # ── Copy to Drive ────────────────────────────────────────
    print("💾 Copying datasets to Drive...")
    for ds in ["dataset1", "dataset2"]:
        src = f"{tmp_dir}/{ds}"
        dst = f"{DATASETS_DIR}/{ds}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        csvs = [f for f in os.listdir(dst) if f.endswith(".csv")]
        print(f"  ✅ {ds}: {csvs}")

    mark_done("datasets_downloaded")
    print("✅ Datasets saved to Drive.")

# ── Create local symlinks for convenience ────────────────────
os.makedirs("/content/data", exist_ok=True)
for ds in ["dataset1", "dataset2"]:
    link = f"/content/data/{ds}"
    if not os.path.exists(link):
        os.symlink(f"{DATASETS_DIR}/{ds}", link)
print("🔗 Symlinks created at /content/data/")
print(f"  dataset1 → {os.listdir('/content/data/dataset1')}")
print(f"  dataset2 → {os.listdir('/content/data/dataset2')}")
```

> ### 🧠 Cell 4 Explanation
>
> This cell **downloads the two phishing datasets from Kaggle** and saves them to Google Drive. Kaggle is a popular platform for machine learning datasets and competitions. To download datasets programmatically (without manually clicking "Download"), you need a Kaggle API key.
>
> **`zipfile`** — Built-in Python module for creating/extracting `.zip` files. Used internally by the Kaggle CLI for unzipping downloads.
>
> **`from google.colab import files`** — Provides `files.upload()`, which opens a file picker in the Colab interface allowing you to upload a file from your computer into the Colab session.
>
> **`_datasets_present()`** — A helper function that checks whether both dataset folders already exist and contain at least one `.csv` file. `os.path.isdir(d1)` checks if a folder exists. `any(f.endswith(".csv") for f in os.listdir(d1))` checks if any file in that folder has a `.csv` extension.
>
> **Checkpoint guard with file verification** — Even if the checkpoint says "downloaded," the cell verifies the files actually exist. If the Drive files somehow got deleted, it resets the checkpoint flag and re-downloads. This is a safety net.
>
> **`files.upload()`** — Triggers a file upload dialog in Colab. You upload your `kaggle.json` file here (this is your personal Kaggle API credentials file downloaded from your Kaggle account settings).
>
> **`os.path.expanduser("~/.kaggle")`** — `expanduser` converts the `~` shorthand to the actual home directory path (e.g., `/root`). The Kaggle CLI looks for credentials in `~/.kaggle/kaggle.json`.
>
> **`os.chmod(kaggle_json_dst, 0o600)`** — Sets strict file permissions on `kaggle.json` so only the current user can read it. The Kaggle CLI requires this for security — it refuses to work if the file is readable by others.
>
> **`os.system(...)`** — Runs a shell command. Here it runs the `kaggle datasets download` command to download each dataset directly to a temporary folder. The `--unzip` flag automatically extracts the downloaded zip file.
>
> **`shutil.rmtree(dst)`** — Deletes an entire folder and everything inside it (like "delete folder" in a file explorer).
>
> **`shutil.copytree(src, dst)`** — Copies an entire folder tree from a source location to a destination. Used to move downloaded data from the temporary `/content/tmp` folder to permanent storage on Google Drive.
>
> **Symlinks** — `os.symlink(source, link)` creates a shortcut (symbolic link). Instead of accessing data through the long Drive path, cells can now use the short `/content/data/dataset1` path, which points to the Drive folder behind the scenes. This is purely a convenience shortcut.

---

## CELL 5 — Load & Inspect Dataset 1 (PhiUSIIL)

```python
# ============================================================
# CELL 5 — Load & Inspect Dataset 1 (PhiUSIIL)
# ============================================================

if is_done("dataset1_loaded"):
    print("✅ Skipping — Dataset 1 already loaded and inspected.")
    if TARGET_COL_1 is None:
        TARGET_COL_1 = load_checkpoint()["metadata"].get("TARGET_COL_1")
    print(f"   TARGET_COL_1 = '{TARGET_COL_1}'")
else:
    # ── Load ────────────────────────────────────────────────
    ds1_dir = "/content/data/dataset1"
    csv_files = [f for f in os.listdir(ds1_dir) if f.endswith(".csv")]
    assert len(csv_files) > 0, "No CSV found in dataset1!"
    df1 = pd.read_csv(os.path.join(ds1_dir, csv_files[0]))

    # Normalize columns
    df1.columns = df1.columns.str.strip().str.lower().str.replace(' ', '_')

    print("=" * 60)
    print(f"  DATASET 1 — PhiUSIIL  |  File: {csv_files[0]}")
    print("=" * 60)
    print(f"\n📐 Shape        : {df1.shape}")
    print(f"\n📋 Dtypes:\n{df1.dtypes.to_string()}")
    print(f"\n🔍 Head (5 rows):\n{df1.head(5).to_string()}")
    print(f"\n📊 Describe:\n{df1.describe().to_string()}")

    null_counts = df1.isnull().sum().sort_values(ascending=False)
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        print(f"\n⚠️  Null Values:\n{null_counts.to_string()}")
    else:
        print("\n✅ No null values found.")

    # ── Auto-detect target column ────────────────────────────
    target_candidates = ['label', 'phishing', 'result', 'status', 'class']
    TARGET_COL_1 = None
    for col in df1.columns:
        if col.lower() in target_candidates:
            TARGET_COL_1 = col
            break
    assert TARGET_COL_1 is not None, (
        f"Could not auto-detect target column! Columns: {df1.columns.tolist()}"
    )
    print(f"\n🎯 Target column detected: '{TARGET_COL_1}'")
    print(f"\n📊 Class distribution:\n{df1[TARGET_COL_1].value_counts().to_string()}")

    # ── Save metadata to checkpoint ──────────────────────────
    _s = load_checkpoint()
    _s["metadata"]["TARGET_COL_1"] = TARGET_COL_1
    _s["metadata"]["dataset1_shape"] = list(df1.shape)
    save_checkpoint(_s)

    mark_done("dataset1_loaded")
    print("\n✅ Dataset 1 loaded and inspected.")
```

> ### 🧠 Cell 5 Explanation
>
> This cell **loads Dataset 1 (PhiUSIIL)** from the CSV file into a pandas DataFrame and performs a first-look inspection of the data.
>
> **`pd.read_csv(filepath)`** — Reads a CSV (Comma-Separated Values) file — basically a plain text file where each row is a data record and columns are separated by commas. Returns a pandas DataFrame (a table) with all the data loaded into memory.
>
> **`os.path.join(ds1_dir, csv_files[0])`** — Safely combines folder path and filename into a full file path. For example, `/content/data/dataset1` + `phiusiil.csv` → `/content/data/dataset1/phiusiil.csv`. Using `os.path.join` instead of simple string concatenation is safer across different operating systems.
>
> **Column normalization** — `df1.columns.str.strip().str.lower().str.replace(' ', '_')` cleans the column names:
> - `.str.strip()` — Removes any leading or trailing spaces from column names.
> - `.str.lower()` — Converts all column names to lowercase so `"URL_Length"` and `"url_length"` are treated the same.
> - `.str.replace(' ', '_')` — Replaces spaces with underscores so column names work well in code (e.g., `"url length"` → `"url_length"`).
>
> **`df1.shape`** — Returns a tuple like `(87,000, 56)` meaning 87,000 rows (URLs) and 56 columns (features).
>
> **`df1.dtypes`** — Shows the data type of each column (integer, float, object/text, etc.). Helps identify which columns are numeric and which are text.
>
> **`df1.head(5)`** — Returns the first 5 rows of the DataFrame. A quick preview to see what the data actually looks like.
>
> **`df1.describe()`** — Generates summary statistics for all numeric columns: count, mean, min, max, quartiles (25%, 50%, 75%). A fast way to understand the range and distribution of each feature.
>
> **`df1.isnull().sum()`** — `isnull()` creates a DataFrame of `True`/`False` values indicating missing data. `.sum()` counts the `True` values per column, telling you how many missing entries each column has. `.sort_values(ascending=False)` orders them from most missing to least.
>
> **Auto-detect target column** — The code searches for column names like `'label'`, `'phishing'`, `'result'`, `'status'`, or `'class'` — common names for the answer column (what the model is trying to predict). `assert` throws an error with a helpful message if no such column is found.
>
> **`value_counts()`** — Counts how many times each unique value appears in a column. For the target column, this tells you how many URLs are labeled "phishing" vs "legitimate" — essential for understanding class imbalance.
>
> **Metadata saving** — The detected target column name and dataset dimensions are saved to the checkpoint so Cell 0's auto-restore can bring them back in future sessions.

---

## CELL 6 — Load & Inspect Dataset 2 (Web Page Phishing)

```python
# ============================================================
# CELL 6 — Load & Inspect Dataset 2 (Web Page Phishing)
# ============================================================

if is_done("dataset2_loaded"):
    print("✅ Skipping — Dataset 2 already loaded and inspected.")
    if TARGET_COL_2 is None:
        TARGET_COL_2 = load_checkpoint()["metadata"].get("TARGET_COL_2")
    print(f"   TARGET_COL_2 = '{TARGET_COL_2}'")
else:
    ds2_dir = "/content/data/dataset2"
    csv_files = [f for f in os.listdir(ds2_dir) if f.endswith(".csv")]
    assert len(csv_files) > 0, "No CSV found in dataset2!"
    df2 = pd.read_csv(os.path.join(ds2_dir, csv_files[0]))

    df2.columns = df2.columns.str.strip().str.lower().str.replace(' ', '_')

    print("=" * 60)
    print(f"  DATASET 2 — Web Page Phishing  |  File: {csv_files[0]}")
    print("=" * 60)
    print(f"\n📐 Shape        : {df2.shape}")
    print(f"\n📋 Dtypes:\n{df2.dtypes.to_string()}")
    print(f"\n🔍 Head (5 rows):\n{df2.head(5).to_string()}")
    print(f"\n📊 Describe:\n{df2.describe().to_string()}")

    null_counts = df2.isnull().sum().sort_values(ascending=False)
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        print(f"\n⚠️  Null Values:\n{null_counts.to_string()}")
    else:
        print("\n✅ No null values found.")

    target_candidates = ['label', 'phishing', 'result', 'status', 'class']
    TARGET_COL_2 = None
    for col in df2.columns:
        if col.lower() in target_candidates:
            TARGET_COL_2 = col
            break
    assert TARGET_COL_2 is not None, (
        f"Could not auto-detect target column! Columns: {df2.columns.tolist()}"
    )
    print(f"\n🎯 Target column detected: '{TARGET_COL_2}'")
    print(f"\n📊 Class distribution:\n{df2[TARGET_COL_2].value_counts().to_string()}")

    _s = load_checkpoint()
    _s["metadata"]["TARGET_COL_2"] = TARGET_COL_2
    _s["metadata"]["dataset2_shape"] = list(df2.shape)
    save_checkpoint(_s)

    mark_done("dataset2_loaded")
    print("\n✅ Dataset 2 loaded and inspected.")
```

> ### 🧠 Cell 6 Explanation
>
> This cell does **exactly the same thing as Cell 5, but for Dataset 2** (the "Web Page Phishing Detection" dataset). The logic, methods, and structure are identical — `pd.read_csv`, column normalization, shape/dtype inspection, null count, target column auto-detection, and checkpoint saving. The only difference is the source folder (`dataset2`) and that results are stored in `df2` and `TARGET_COL_2` instead of `df1` and `TARGET_COL_1`.
>
> The reason two separate datasets are used is to make the research more robust — a model that works well on both independent datasets is more convincingly reliable than one tested on only one source.

---

## CELL 7 — EDA — Dataset 1

```python
# ============================================================
# CELL 7 — EDA — Dataset 1
# ============================================================

from IPython.display import Image as IPyImage, display

if is_done("eda_done"):
    print("✅ EDA (DS1) already done — displaying saved plots.")
    for fname in sorted(os.listdir(PLOTS_DIR)):
        if fname.startswith("eda_dataset1_"):
            display(IPyImage(filename=os.path.join(PLOTS_DIR, fname)))
else:
    if 'df1' not in dir():
        ds1_dir = "/content/data/dataset1"
        csv_files = [f for f in os.listdir(ds1_dir) if f.endswith(".csv")]
        df1 = pd.read_csv(os.path.join(ds1_dir, csv_files[0]))
        df1.columns = df1.columns.str.strip().str.lower().str.replace(' ', '_')
    if TARGET_COL_1 is None:
        TARGET_COL_1 = load_checkpoint()["metadata"]["TARGET_COL_1"]

    numeric_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL_1 in numeric_cols:
        numeric_cols.remove(TARGET_COL_1)

    # ── Plot 1: Class Distribution ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    vc = df1[TARGET_COL_1].value_counts()
    bars = sns.countplot(x=TARGET_COL_1, data=df1, palette='Set2', ax=ax)
    for p in ax.patches:
        count = int(p.get_height())
        pct = 100 * count / len(df1)
        ax.annotate(f'{count:,}\n({pct:.1f}%)', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=11)
    ax.set_title("Class Distribution — PhiUSIIL Dataset", fontsize=14, fontweight='bold')
    ax.set_xlabel("Class (0=Legit, 1=Phishing)"); ax.set_ylabel("Count")
    plt.tight_layout()
    p1_path = f"{PLOTS_DIR}/eda_dataset1_class_dist.png"
    plt.savefig(p1_path, dpi=300); plt.show(); plt.close()

    # ── Plot 2: Missing Values Heatmap ───────────────────────
    if df1.isnull().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(df1.sample(1000, random_state=RANDOM_SEED).isnull(),
                    yticklabels=False, cbar=False, cmap='viridis', ax=ax)
        ax.set_title("Missing Values Heatmap (1000-row sample) — DS1", fontsize=13, fontweight='bold')
        plt.tight_layout()
        p2_path = f"{PLOTS_DIR}/eda_dataset1_missing.png"
        plt.savefig(p2_path, dpi=300); plt.show(); plt.close()
    else:
        print("✅ Plot 2 skipped — no missing values.")

    # ── Plot 3: Top 20 Feature Correlation Heatmap ───────────
    corr_with_target = df1[numeric_cols + [TARGET_COL_1]].corr()[TARGET_COL_1].drop(TARGET_COL_1)
    top20 = corr_with_target.abs().nlargest(20).index.tolist()
    corr_matrix = df1[top20].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='RdYlGn', ax=ax,
                linewidths=0.3, square=True)
    ax.set_title("Top 20 Feature Correlation Heatmap — DS1", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p3_path = f"{PLOTS_DIR}/eda_dataset1_corr_heatmap.png"
    plt.savefig(p3_path, dpi=300); plt.show(); plt.close()

    # ── Plot 4: KDE for URL length & domain length ───────────
    kde_cols = [c for c in numeric_cols if 'url' in c or 'length' in c or 'domain' in c][:2]
    if len(kde_cols) < 2:
        kde_cols = numeric_cols[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, kde_cols):
        for cls in sorted(df1[TARGET_COL_1].unique()):
            subset = df1[df1[TARGET_COL_1] == cls][col].dropna()
            subset.plot.kde(ax=ax, label=f"Class {cls}", linewidth=2)
        ax.set_title(f"KDE: {col}", fontsize=12); ax.legend()
    plt.suptitle("Feature Distribution by Class — DS1", fontsize=14, fontweight='bold')
    plt.tight_layout()
    p4_path = f"{PLOTS_DIR}/eda_dataset1_kde.png"
    plt.savefig(p4_path, dpi=300); plt.show(); plt.close()

    # ── Plot 5: Boxplot of Top 5 Correlated Features ─────────
    top5 = corr_with_target.abs().nlargest(5).index.tolist()
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, col in zip(axes, top5):
        sns.boxplot(x=TARGET_COL_1, y=col, data=df1, palette='Set3', ax=ax)
        ax.set_title(col[:20], fontsize=10)
    plt.suptitle("Boxplot — Top 5 Correlated Features by Class — DS1",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p5_path = f"{PLOTS_DIR}/eda_dataset1_boxplots.png"
    plt.savefig(p5_path, dpi=300); plt.show(); plt.close()

    # ── Plot 6: Pairplot (2000-row sample, top 4 features) ───
    top4 = corr_with_target.abs().nlargest(4).index.tolist()
    sample_df = df1[top4 + [TARGET_COL_1]].sample(2000, random_state=RANDOM_SEED)
    sample_df[TARGET_COL_1] = sample_df[TARGET_COL_1].astype(str)
    pair_fig = sns.pairplot(sample_df, hue=TARGET_COL_1, diag_kind='kde',
                            plot_kws={"alpha": 0.4}, palette='Set1')
    pair_fig.fig.suptitle("Pairplot — Top 4 Features (2000-sample) — DS1",
                          y=1.02, fontsize=13, fontweight='bold')
    p6_path = f"{PLOTS_DIR}/eda_dataset1_pairplot.png"
    pair_fig.savefig(p6_path, dpi=300)
    plt.show(); plt.close()

    mark_done("eda_done")
    print("\n✅ EDA for Dataset 1 complete. All 6 plots saved to Drive.")
```

> ### 🧠 Cell 7 Explanation
>
> This cell performs **Exploratory Data Analysis (EDA)** on Dataset 1. EDA means visually and statistically exploring your data to understand its structure, patterns, and problems before modelling. Think of it as "getting to know your data."
>
> **`from IPython.display import Image as IPyImage, display`** — `IPyImage` is used to render saved image files (`.png`) directly inside the Colab notebook output. `display()` triggers the in-notebook rendering.
>
> **`df1.select_dtypes(include=[np.number])`** — Filters the DataFrame to keep only columns whose values are numbers (integers or floats). Text/string columns are excluded because many statistical operations only work on numbers.
>
> **Plot 1 — Class Distribution Bar Chart:**
> - `plt.subplots(figsize=(7,5))` — Creates a blank figure (canvas) that is 7 inches wide and 5 inches tall. Returns two objects: `fig` (the whole figure) and `ax` (the drawing area inside it).
> - `sns.countplot(x=TARGET_COL_1, data=df1, palette='Set2', ax=ax)` — Draws a bar chart where each bar represents one class (0=Legit, 1=Phishing) and the bar height is the count of rows in that class. `palette='Set2'` applies a pleasant color scheme.
> - `ax.patches` — Each bar in the plot is a "patch" (a drawn rectangle). The loop annotates each bar with its count and percentage.
> - `ax.annotate(...)` — Writes text at a specific position on the chart. Used to label bars with count and percentage.
> - `plt.savefig(p1_path, dpi=300)` — Saves the chart to a PNG file at 300 DPI (high resolution, suitable for publications). `plt.close()` frees memory by closing the figure.
>
> **Plot 2 — Missing Values Heatmap:**
> - `df1.sample(1000, random_state=RANDOM_SEED)` — Randomly picks 1,000 rows from the dataset. The full dataset would be too slow to render.
> - `df1.isnull()` — Creates a grid of True/False values; True = the cell is missing/empty.
> - `sns.heatmap(...)` — Draws a color grid where each cell's color shows whether a value is missing. Dark = missing, light = present.
>
> **Plot 3 — Correlation Heatmap:**
> - `.corr()` — Computes how strongly each pair of columns is correlated (values from -1 to +1). +1 = perfectly move together, -1 = perfectly move opposite, 0 = no relationship.
> - `corr_with_target.abs().nlargest(20)` — Takes the top 20 features most strongly correlated (positively or negatively) with the target column.
> - `cmap='RdYlGn'` — Red-Yellow-Green colormap: green = strong positive correlation, red = strong negative correlation.
>
> **Plot 4 — KDE (Kernel Density Estimation) Plots:**
> - KDE is a smooth curve showing the distribution of values for a feature — like a smoothed histogram. Showing one curve per class (phishing vs. legit) reveals whether the feature's values differ between the two classes. If the curves overlap a lot, the feature is less useful. If they're clearly separated, it's a strong predictor.
> - `subset.plot.kde(ax=ax, ...)` — Draws the KDE curve for a specific subset of data.
>
> **Plot 5 — Boxplots:**
> - A boxplot shows the median, quartiles, and outliers for a feature, split by class. `sns.boxplot(x=TARGET_COL_1, y=col, ...)` draws side-by-side boxplots for phishing vs. legit, making it easy to see if the distributions are different.
>
> **Plot 6 — Pairplot:**
> - A pairplot shows scatter plots for every combination of the top 4 features, with diagonal plots showing each feature's distribution. `hue=TARGET_COL_1` colors points by class. `alpha=0.4` makes points semi-transparent to reduce clutter where they overlap. This helps visualize if the two classes are separable in the feature space.
> - `sample(2000)` — Only 2,000 rows are used because plotting all rows would be extremely slow.

---

## CELL 8 — EDA — Dataset 2

```python
# ============================================================
# CELL 8 — EDA — Dataset 2
# ============================================================

from IPython.display import Image as IPyImage, display

if is_done("eda_ds2_done"):
    print("✅ EDA (DS2) already done — displaying saved plots.")
    for fname in sorted(os.listdir(PLOTS_DIR)):
        if fname.startswith("eda_dataset2_"):
            display(IPyImage(filename=os.path.join(PLOTS_DIR, fname)))
else:
    if 'df2' not in dir():
        ds2_dir = "/content/data/dataset2"
        csv_files = [f for f in os.listdir(ds2_dir) if f.endswith(".csv")]
        df2 = pd.read_csv(os.path.join(ds2_dir, csv_files[0]))
        df2.columns = df2.columns.str.strip().str.lower().str.replace(' ', '_')
    if TARGET_COL_2 is None:
        TARGET_COL_2 = load_checkpoint()["metadata"]["TARGET_COL_2"]

    numeric_cols2 = df2.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL_2 in numeric_cols2:
        numeric_cols2.remove(TARGET_COL_2)

    # ── Plot 1: Class Distribution ───────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=TARGET_COL_2, data=df2, palette='Set2', ax=ax)
    for p in ax.patches:
        count = int(p.get_height())
        pct = 100 * count / len(df2)
        ax.annotate(f'{count:,}\n({pct:.1f}%)', (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=11)
    ax.set_title("Class Distribution — Web Page Phishing Dataset", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p1_path = f"{PLOTS_DIR}/eda_dataset2_class_dist.png"
    plt.savefig(p1_path, dpi=300); plt.show(); plt.close()

    # ── Plot 2: Feature Importance Preview (quick RF) ────────
    from sklearn.ensemble import RandomForestClassifier as _RFC
    _X = df2[numeric_cols2].fillna(0)
    _y = df2[TARGET_COL_2]
    _rf_quick = _RFC(n_estimators=50, random_state=RANDOM_SEED, n_jobs=-1)
    _rf_quick.fit(_X, _y)
    _fi = pd.Series(_rf_quick.feature_importances_, index=numeric_cols2).nlargest(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    _fi.sort_values().plot.barh(ax=ax, color='steelblue')
    ax.set_title("Feature Importance Preview (Quick RF) — DS2", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance")
    plt.tight_layout()
    p2_path = f"{PLOTS_DIR}/eda_dataset2_feature_importance.png"
    plt.savefig(p2_path, dpi=300); plt.show(); plt.close()

    # ── Plot 3: Correlation Heatmap ──────────────────────────
    corr2 = df2[numeric_cols2[:20]].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr2, annot=False, cmap='RdYlGn', ax=ax, linewidths=0.3)
    ax.set_title("Correlation Heatmap — DS2 (top 20 numeric features)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    p3_path = f"{PLOTS_DIR}/eda_dataset2_corr_heatmap.png"
    plt.savefig(p3_path, dpi=300); plt.show(); plt.close()

    mark_done("eda_ds2_done")
    print("\n✅ EDA for Dataset 2 complete. 3 plots saved to Drive.")
```

> ### 🧠 Cell 8 Explanation
>
> This cell performs **EDA on Dataset 2**, similar to Cell 7 but with a slightly different set of plots tailored for this dataset.
>
> **Plot 1 — Class Distribution** works identically to Cell 7's Plot 1.
>
> **Plot 2 — Quick Feature Importance using a small Random Forest:**
> - A "quick" Random Forest with only `n_estimators=50` (50 trees, instead of the full 200 used in training) is trained purely for EDA purposes to get a rough idea of which features are most useful.
> - `.fillna(0)` — Fills any missing values with 0 before training (the quick RF doesn't need perfectly cleaned data).
> - `_rf_quick.feature_importances_` — After training, a Random Forest can tell you how much each feature contributed to making accurate splits across all trees. Higher = more important.
> - `pd.Series(...).nlargest(20)` — Creates a pandas Series from the importance scores and picks the top 20.
> - `.plot.barh(ax=ax, color='steelblue')` — Draws a horizontal bar chart. `barh` means "bar horizontal" — useful when feature names are long.
> - `n_jobs=-1` — Tells scikit-learn to use all available CPU cores in parallel to speed up training.
>
> **Plot 3 — Correlation Heatmap** works the same as in Cell 7 but uses only the first 20 numeric columns (`numeric_cols2[:20]`) to keep the chart readable.

---

## CELL 9 — Preprocessing — Dataset 1

```python
# ============================================================
# CELL 9 — Preprocessing — Dataset 1
# ============================================================

if is_done("preprocessing_done"):
    print("✅ Preprocessed splits already on Drive — skipping.")
    print(f"   X1_train: {X1_train.shape}, X1_test: {X1_test.shape}")
else:
    if 'df1' not in dir():
        ds1_dir = "/content/data/dataset1"
        csv_files = [f for f in os.listdir(ds1_dir) if f.endswith(".csv")]
        df1 = pd.read_csv(os.path.join(ds1_dir, csv_files[0]))
        df1.columns = df1.columns.str.strip().str.lower().str.replace(' ', '_')
    if TARGET_COL_1 is None:
        TARGET_COL_1 = load_checkpoint()["metadata"]["TARGET_COL_1"]

    print("🔧 Preprocessing Dataset 1...")
    df = df1.copy()

    # Step 1: Drop URL/ID cols and high-null cols
    url_id_cols = [c for c in df.columns if any(k in c for k in ['url', 'id', 'index'])]
    df.drop(columns=url_id_cols, errors='ignore', inplace=True)
    high_null = [c for c in df.columns if df[c].isnull().mean() > 0.5]
    df.drop(columns=high_null, errors='ignore', inplace=True)

    # Step 2: Fill nulls
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # Step 3: Encode target to binary
    le_target = LabelEncoder()
    df[TARGET_COL_1] = le_target.fit_transform(df[TARGET_COL_1])
    label_mapping = {str(cls): int(enc) for cls, enc in
                     zip(le_target.classes_, le_target.transform(le_target.classes_))}
    save_to_drive(label_mapping, "metrics/label_mapping.json")

    # Step 4: Encode remaining categoricals
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL_1]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Step 5: Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)

    # Step 6-7: Split
    X1 = df.drop(columns=[TARGET_COL_1]).astype(np.float32)
    y1 = df[TARGET_COL_1].astype(int)
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        X1, y1, test_size=0.2, random_state=RANDOM_SEED, stratify=y1
    )

    # Step 8: Scale
    scaler1 = StandardScaler()
    X1_train = pd.DataFrame(scaler1.fit_transform(X1_train), columns=X1.columns)
    X1_test  = pd.DataFrame(scaler1.transform(X1_test),      columns=X1.columns)

    # Save all to Drive
    for obj, rel_path in [
        (X1_train, "checkpoints/preprocessed_X1_train.pkl"),
        (X1_test,  "checkpoints/preprocessed_X1_test.pkl"),
        (y1_train, "checkpoints/preprocessed_y1_train.pkl"),
        (y1_test,  "checkpoints/preprocessed_y1_test.pkl"),
        (scaler1,  "checkpoints/scaler1.pkl"),
    ]:
        save_to_drive(obj, rel_path)

    mark_done("preprocessing_done")
    print("\n✅ Preprocessing DS1 complete. All splits saved to Drive.")
```

> ### 🧠 Cell 9 Explanation
>
> This cell **cleans and prepares Dataset 1** for machine learning. Raw data is almost never ready to feed directly into a model — it needs cleaning and transformation first. This entire pipeline is known as "data preprocessing."
>
> **`df = df1.copy()`** — Creates a full copy of the original DataFrame so that changes don't modify `df1`. This preserves the raw data in case you need it again later.
>
> **Step 1 — Drop useless columns:**
> - URL columns (like the actual URL string) and ID columns (like row numbers) are dropped because they have no predictive value — every URL is unique, so the model can't learn a pattern from raw URLs.
> - `df[c].isnull().mean() > 0.5` — If more than 50% of a column's values are missing, the column is too sparse to be useful and is dropped. `.mean()` on a boolean series gives the fraction of `True` values.
> - `df.drop(columns=..., errors='ignore', inplace=True)` — Removes specified columns. `errors='ignore'` prevents an error if a column name doesn't exist. `inplace=True` modifies `df` directly rather than returning a new copy.
>
> **Step 2 — Fill missing values (imputation):**
> - For numeric columns: missing values are replaced with the **median** of that column. The median (middle value) is preferred over the mean because it's not skewed by extreme outliers.
> - For text/categorical columns: missing values are replaced with the **mode** (most frequent value). `df[col].mode()[0]` returns the single most common value.
>
> **Step 3 — Encode the target column:**
> - `LabelEncoder().fit_transform(df[TARGET_COL_1])` converts text labels like "phishing"/"legitimate" to numbers like 1/0. `fit_transform` = learn the mapping (`fit`) and apply it (`transform`) in one step.
> - `label_mapping` records which text label became which number, saved to Drive for reference.
>
> **Step 4 — Encode other text columns:**
> - `df.select_dtypes(include=['object', 'category'])` selects columns that are text or categorical.
> - Each such column is converted to numbers using `LabelEncoder`. Machine learning models require all inputs to be numbers.
>
> **Step 5 — Drop duplicate rows:**
> - `df.drop_duplicates(inplace=True)` removes any rows that are completely identical to another row. Duplicates can bias the model by making it see the same example multiple times.
>
> **Step 6–7 — Train/Test Split:**
> - `X1 = df.drop(columns=[TARGET_COL_1])` — The features (inputs) are everything except the target column.
> - `y1 = df[TARGET_COL_1]` — The labels (what we're trying to predict).
> - `train_test_split(X1, y1, test_size=0.2, stratify=y1)` — Splits data into 80% training and 20% testing. `stratify=y1` ensures both splits have the same proportion of phishing vs. legitimate URLs — important for imbalanced datasets.
>
> **Step 8 — Feature Scaling:**
> - `StandardScaler()` standardizes features so each has mean 0 and standard deviation 1. This prevents features with large numerical ranges (like URL length = 200) from dominating features with small ranges (like number of subdomains = 2).
> - `scaler1.fit_transform(X1_train)` — Learns the mean/std from the training data and applies scaling. `fit` = learn, `transform` = apply.
> - `scaler1.transform(X1_test)` — Applies the **same** scaling (learned from training data) to test data. **Critical:** the test data uses the training scaler, not its own, to prevent "data leakage" (accidentally using test data statistics during training).
> - Everything is wrapped back into a `pd.DataFrame` with column names preserved so downstream code can reference features by name.
>
> **All preprocessed splits are saved to Drive** as `.pkl` files so they can be reloaded in future sessions without re-running this entire pipeline.

---

## CELL 10 — Preprocessing — Dataset 2

```python
# ============================================================
# CELL 10 — Preprocessing — Dataset 2
# ============================================================

if is_done("preprocessing_ds2_done"):
    print("✅ Preprocessed DS2 splits already on Drive — skipping.")
else:
    if 'df2' not in dir():
        ds2_dir = "/content/data/dataset2"
        csv_files = [f for f in os.listdir(ds2_dir) if f.endswith(".csv")]
        df2 = pd.read_csv(os.path.join(ds2_dir, csv_files[0]))
        df2.columns = df2.columns.str.strip().str.lower().str.replace(' ', '_')
    if TARGET_COL_2 is None:
        TARGET_COL_2 = load_checkpoint()["metadata"]["TARGET_COL_2"]

    df = df2.copy()

    url_id_cols = [c for c in df.columns if any(k in c for k in ['url', 'id', 'index'])]
    df.drop(columns=url_id_cols, errors='ignore', inplace=True)
    high_null = [c for c in df.columns if df[c].isnull().mean() > 0.5]
    df.drop(columns=high_null, errors='ignore', inplace=True)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    le_target2 = LabelEncoder()
    df[TARGET_COL_2] = le_target2.fit_transform(df[TARGET_COL_2])

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL_2]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    before = len(df)
    df.drop_duplicates(inplace=True)

    X2 = df.drop(columns=[TARGET_COL_2]).astype(np.float32)
    y2 = df[TARGET_COL_2].astype(int)
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=RANDOM_SEED, stratify=y2
    )

    # Skip scaling if features already in [0,1]
    feat_range = X2_train.max().max() - X2_train.min().min()
    if feat_range > 2.0:
        scaler2 = StandardScaler()
        X2_train = pd.DataFrame(scaler2.fit_transform(X2_train), columns=X2.columns)
        X2_test  = pd.DataFrame(scaler2.transform(X2_test),      columns=X2.columns)
    else:
        scaler2 = None
        print("  ℹ️  Features already in [0,1] — scaling skipped for DS2")

    for obj, rel_path in [
        (X2_train, "checkpoints/preprocessed_X2_train.pkl"),
        (X2_test,  "checkpoints/preprocessed_X2_test.pkl"),
        (y2_train, "checkpoints/preprocessed_y2_train.pkl"),
        (y2_test,  "checkpoints/preprocessed_y2_test.pkl"),
        (scaler2,  "checkpoints/scaler2.pkl"),
    ]:
        save_to_drive(obj, rel_path)

    mark_done("preprocessing_ds2_done")
    print("\n✅ Preprocessing DS2 complete. All splits saved to Drive.")
```

> ### 🧠 Cell 10 Explanation
>
> This cell runs the **same preprocessing pipeline as Cell 9 but for Dataset 2.** All steps — drop URL/ID columns, fill nulls, encode labels, encode categoricals, drop duplicates, train/test split — are identical.
>
> **One key difference — Smart Scaling Check:**
> - `feat_range = X2_train.max().max() - X2_train.min().min()` — Computes the overall range of values across all features. `.max().max()` first finds the max of each column, then finds the max of those maxes.
> - `if feat_range > 2.0` — If the feature values span more than 2 units, scaling is applied. If features are already in the range [0, 1] (which is common for datasets whose features were pre-normalized), scaling would actually shrink and distort the values, so it's skipped.
> - `scaler2 = None` is set when skipping, so downstream code can check if scaling was applied.
>
> This smart check prevents the scaling step from actually hurting the data quality on datasets that are already properly scaled.

---

## CELL 11 — SMOTE (Class Imbalance Handling)

```python
# ============================================================
# CELL 11 — SMOTE (Class Imbalance Handling)
# ============================================================

if is_done("smote_done"):
    print("✅ SMOTE already applied — loading from Drive.")
    X1_train_sm = load_from_drive("checkpoints/X1_train_smote.pkl")
    y1_train_sm = load_from_drive("checkpoints/y1_train_smote.pkl")
else:
    vc = pd.Series(y1_train).value_counts()
    majority = vc.iloc[0]
    minority = vc.iloc[-1]
    ratio = majority / minority

    if ratio > 1.5:
        print("⚙️  Applying SMOTE...")
        smote = SMOTE(random_state=RANDOM_SEED)
        X1_arr = X1_train.values if hasattr(X1_train, 'values') else X1_train
        X1_sm_arr, y1_train_sm = smote.fit_resample(X1_arr, y1_train)
        X1_train_sm = pd.DataFrame(X1_sm_arr, columns=X1_train.columns)
    else:
        print("ℹ️  Imbalance ratio ≤ 1.5 — SMOTE skipped.")
        X1_train_sm = X1_train.copy()
        y1_train_sm = y1_train.copy()

    save_to_drive(X1_train_sm, "checkpoints/X1_train_smote.pkl")
    save_to_drive(y1_train_sm, "checkpoints/y1_train_smote.pkl")

    mark_done("smote_done")
    print("\n✅ SMOTE step complete. Saved to Drive.")
```

> ### 🧠 Cell 11 Explanation
>
> This cell handles **class imbalance** — a situation where one class (e.g., legitimate URLs) has far more examples than the other (phishing URLs). If left unaddressed, a model could achieve high accuracy by simply predicting "legitimate" for everything.
>
> **Why imbalance is a problem:** Imagine 95% of URLs are legitimate and 5% are phishing. A lazy model that always says "legitimate" achieves 95% accuracy but catches zero phishing URLs — completely useless for our purpose.
>
> **`pd.Series(y1_train).value_counts()`** — Counts how many training examples belong to each class.
> - `vc.iloc[0]` — The count of the most common class (majority).
> - `vc.iloc[-1]` — The count of the least common class (minority).
> - `ratio = majority / minority` — If this is, say, 5.0, it means there are 5x more legitimate URLs than phishing ones in the training data.
>
> **SMOTE — Synthetic Minority Over-sampling Technique:**
> - `SMOTE(random_state=RANDOM_SEED)` — Creates an instance of the SMOTE algorithm.
> - `smote.fit_resample(X1_arr, y1_train)` — Generates new synthetic phishing examples by interpolating between existing phishing examples. Specifically, for each minority-class point, SMOTE finds its nearest neighbors and creates new fake-but-realistic points along the line between them. The result is a balanced training set where both classes have equal counts.
> - SMOTE is only applied if the imbalance ratio is greater than 1.5 (i.e., one class has more than 50% more examples than the other). If the dataset is roughly balanced, SMOTE is skipped.
>
> **`X1_train.values`** — `.values` converts a pandas DataFrame to a plain numpy array, required because SMOTE accepts arrays, not DataFrames.
>
> **`pd.DataFrame(X1_sm_arr, columns=X1_train.columns)`** — Converts the result back to a DataFrame with the original column names preserved.
>
> **Important:** SMOTE is only applied to the **training** data, never the test data. The test set must represent the real-world distribution for evaluation to be meaningful.

---

## CELL 12 — Model 1: Logistic Regression

```python
# ============================================================
# CELL 12 — Model 1: Logistic Regression
# ============================================================

if is_done("lr_trained"):
    print("✅ LR already trained — loading from Drive.")
    lr_model   = load_from_drive("models/lr_model.pkl")
    lr_metrics = load_from_drive("metrics/lr_metrics.json")
else:
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)

    t0 = time.time()
    lr_model.fit(X1_train_sm, y1_train_sm)
    lr_train_time = time.time() - t0

    t1 = time.time()
    y_pred_lr = lr_model.predict(X1_test)
    lr_infer_time = (time.time() - t1) / len(X1_test) * 1000

    y_prob_lr = lr_model.predict_proba(X1_test)[:, 1]

    lr_metrics = {
        "accuracy":   accuracy_score(y1_test, y_pred_lr),
        "precision":  precision_score(y1_test, y_pred_lr, zero_division=0),
        "recall":     recall_score(y1_test, y_pred_lr, zero_division=0),
        "f1":         f1_score(y1_test, y_pred_lr, zero_division=0),
        "roc_auc":    roc_auc_score(y1_test, y_prob_lr),
        "train_time": round(lr_train_time, 4),
        "infer_time_ms": round(lr_infer_time, 6),
    }

    print("\n📊 Classification Report (LR):")
    print(classification_report(y1_test, y_pred_lr))

    save_to_drive(lr_model,   "models/lr_model.pkl")
    save_to_drive(lr_metrics, "metrics/lr_metrics.json")

    mark_done("lr_trained")
    print("\n✅ LR model saved to Drive.")
```

> ### 🧠 Cell 12 Explanation
>
> This cell **trains the first model: Logistic Regression**, which serves as a baseline — the simplest model to beat.
>
> **What is Logistic Regression?** Despite the word "regression," it's a classification algorithm. It learns a mathematical boundary (a line or hyperplane) that separates phishing URLs from legitimate ones. It outputs a probability between 0 and 1 for each URL.
>
> **`LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)`**
> - `max_iter=1000` — The algorithm is iterative (it improves its estimates step by step). This sets the maximum number of steps before stopping. More iterations = better convergence for complex data.
> - `random_state=RANDOM_SEED` — Makes the model training reproducible.
> - `n_jobs=-1` — Uses all CPU cores for parallel computation.
>
> **`lr_model.fit(X1_train_sm, y1_train_sm)`** — Trains the model on the SMOTE-balanced training data. "fit" means the model learns the relationship between features and labels by adjusting its internal parameters.
>
> **`time.time()`** — Returns the current time as a floating-point number of seconds since a fixed reference point. The training time is computed as `time.time() - t0` (the difference between start and end times). Same for inference time.
>
> **`lr_model.predict(X1_test)`** — Makes binary predictions (0 or 1) for the test set. Returns the most likely class for each URL.
>
> **`lr_infer_time = (time.time() - t1) / len(X1_test) * 1000`** — Converts total inference time to milliseconds per sample. Dividing by the number of samples gives per-URL latency, then multiplied by 1000 to get milliseconds.
>
> **`lr_model.predict_proba(X1_test)[:, 1]`** — Returns the probability of being class 1 (phishing) for each test URL. `[:, 1]` selects the second column (probability of class 1) from the result matrix.
>
> **Metrics computed:**
> - `accuracy_score(y1_test, y_pred_lr)` — Fraction of correct predictions.
> - `precision_score` — Of predicted phishing, how many were truly phishing?
> - `recall_score` — Of actual phishing, how many did we detect?
> - `f1_score` — Harmonic mean of precision and recall.
> - `roc_auc_score` — Computed using probabilities, not binary predictions. Measures the model's ranking ability.
> - `zero_division=0` — If a class has no predicted samples, return 0 instead of raising an error.
>
> **`classification_report(y1_test, y_pred_lr)`** — Prints a formatted table with precision, recall, F1, and support (number of samples) for each class.
>
> Model and metrics are saved to Drive for use in later cells.

---

## CELL 13 — Model 2: Random Forest

```python
# ============================================================
# CELL 13 — Model 2: Random Forest
# ============================================================

if is_done("rf_trained"):
    print("✅ RF already trained — loading from Drive.")
    rf_model   = load_from_drive("models/rf_model.pkl")
    rf_metrics = load_from_drive("metrics/rf_metrics.json")
else:
    rf_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)

    t0 = time.time()
    rf_model.fit(X1_train_sm, y1_train_sm)
    rf_train_time = time.time() - t0

    t1 = time.time()
    y_pred_rf = rf_model.predict(X1_test)
    rf_infer_time = (time.time() - t1) / len(X1_test) * 1000

    y_prob_rf = rf_model.predict_proba(X1_test)[:, 1]

    rf_metrics = {
        "accuracy":   accuracy_score(y1_test, y_pred_rf),
        "precision":  precision_score(y1_test, y_pred_rf, zero_division=0),
        "recall":     recall_score(y1_test, y_pred_rf, zero_division=0),
        "f1":         f1_score(y1_test, y_pred_rf, zero_division=0),
        "roc_auc":    roc_auc_score(y1_test, y_prob_rf),
        "train_time": round(rf_train_time, 4),
        "infer_time_ms": round(rf_infer_time, 6),
    }

    rf_feature_importances = pd.Series(
        rf_model.feature_importances_, index=X1_train_sm.columns
    ).sort_values(ascending=False)

    print("\n📊 Classification Report (RF):")
    print(classification_report(y1_test, y_pred_rf))
    print(f"\n🏅 Top 10 Feature Importances:")
    print(rf_feature_importances.head(10).to_string())

    save_to_drive(rf_model,   "models/rf_model.pkl")
    save_to_drive(rf_metrics, "metrics/rf_metrics.json")
    rf_feature_importances.to_csv(
        f"{METRICS_DIR}/rf_feature_importances.csv", header=["importance"]
    )

    mark_done("rf_trained")
    print("\n✅ RF model saved to Drive.")
```

> ### 🧠 Cell 13 Explanation
>
> This cell **trains the second model: Random Forest**, a powerful ensemble algorithm that typically outperforms Logistic Regression on complex tabular data.
>
> **What is a Random Forest?** It builds many decision trees (200 in this case), each trained on a random subset of the data and a random subset of features. When predicting, all trees "vote" and the majority class wins. The randomness prevents any single tree from overfitting (memorizing the training data) and the voting makes the combined result robust.
>
> **`RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)`**
> - `n_estimators=200` — Build 200 decision trees. More trees = more stable results but slower training.
> - `n_jobs=-1` — Train trees in parallel using all CPU cores.
>
> **Training, prediction, and metrics** work identically to Cell 12.
>
> **Feature Importances — a key advantage of Random Forests:**
> - `rf_model.feature_importances_` — After training, the model stores how much each feature reduced impurity (helped make correct splits) across all trees. This gives a built-in ranking of which URL features matter most for detecting phishing.
> - `pd.Series(rf_model.feature_importances_, index=X1_train_sm.columns).sort_values(ascending=False)` — Pairs each importance score with its feature name and sorts from most to least important.
> - `rf_feature_importances.head(10)` — Shows the top 10 most important features.
> - `.to_csv(...)` — Saves the full importance table as a CSV file to Drive for use in the visualization cell.

---

## CELL 14 — Model 3: XGBoost

```python
# ============================================================
# CELL 14 — Model 3: XGBoost (T4 GPU)
# ============================================================

if is_done("xgb_trained"):
    print("✅ XGBoost already trained — loading from Drive.")
    xgb_model   = load_from_drive("models/xgb_model.pkl")
    xgb_metrics = load_from_drive("metrics/xgb_metrics.json")
else:
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=RANDOM_SEED,
        tree_method='hist',
        device='cuda',
        verbosity=0,
    )

    X1_train_sm_arr = X1_train_sm.values if hasattr(X1_train_sm, 'values') else X1_train_sm
    X1_test_arr     = X1_test.values     if hasattr(X1_test,     'values') else X1_test

    t0 = time.time()
    xgb_model.fit(
        X1_train_sm_arr, y1_train_sm,
        eval_set=[(X1_test_arr, y1_test)],
        verbose=50
    )
    xgb_train_time = time.time() - t0

    t1 = time.time()
    y_pred_xgb = xgb_model.predict(X1_test_arr)
    xgb_infer_time = (time.time() - t1) / len(X1_test_arr) * 1000

    y_prob_xgb = xgb_model.predict_proba(X1_test_arr)[:, 1]

    xgb_metrics = {
        "accuracy":   accuracy_score(y1_test, y_pred_xgb),
        "precision":  precision_score(y1_test, y_pred_xgb, zero_division=0),
        "recall":     recall_score(y1_test, y_pred_xgb, zero_division=0),
        "f1":         f1_score(y1_test, y_pred_xgb, zero_division=0),
        "roc_auc":    roc_auc_score(y1_test, y_prob_xgb),
        "train_time": round(xgb_train_time, 4),
        "infer_time_ms": round(xgb_infer_time, 6),
    }

    print("\n📊 Classification Report (XGBoost):")
    print(classification_report(y1_test, y_pred_xgb))

    save_to_drive(xgb_model,   "models/xgb_model.pkl")
    save_to_drive(xgb_metrics, "metrics/xgb_metrics.json")

    mark_done("xgb_trained")
    print("\n✅ XGBoost model saved to Drive.")
```

> ### 🧠 Cell 14 Explanation
>
> This cell **trains the third model: XGBoost (Extreme Gradient Boosting)**, which is known for being one of the most accurate and fastest algorithms on structured/tabular data. This model also leverages the T4 GPU for significantly faster training.
>
> **What is XGBoost?** Like Random Forest, it builds many decision trees. But instead of building them independently and voting, it builds them *sequentially* — each new tree learns to correct the mistakes of the previous trees. This technique is called "gradient boosting" because it uses gradient descent (a mathematical optimization method) to minimize errors step by step.
>
> **Key hyperparameters:**
> - `n_estimators=300` — Build 300 sequential trees.
> - `learning_rate=0.1` — How much each new tree corrects the previous error. Lower = slower but more precise learning. Common values: 0.01–0.3.
> - `max_depth=6` — Maximum depth of each tree (how many questions each tree can ask). Deeper trees can model more complex patterns but also risk memorizing the training data.
> - `subsample=0.8` — Each tree is trained on only 80% of the training data (randomly sampled). This adds randomness and helps prevent overfitting.
> - `colsample_bytree=0.8` — Each tree uses only 80% of the features (randomly selected). Also helps prevent overfitting.
> - `eval_metric='logloss'` — The metric used to evaluate model quality during training. Log loss penalizes confident wrong predictions heavily.
> - `tree_method='hist'` — Uses a histogram-based algorithm for faster tree building.
> - `device='cuda'` — Tells XGBoost to use the GPU (CUDA) for training, which can be 5–20x faster than CPU.
> - `verbosity=0` — Suppresses XGBoost's internal logs.
>
> **`eval_set=[(X1_test_arr, y1_test)], verbose=50`** — Passes test data so XGBoost can monitor performance on unseen data during training. `verbose=50` prints the evaluation metric every 50 rounds, allowing you to see if the model is improving.
>
> **`.values`** — XGBoost works best with numpy arrays. `.values` converts the DataFrame to a numpy array. `hasattr(X1_train_sm, 'values')` checks if the object is a DataFrame (DataFrames have a `.values` attribute) before converting.

---

## CELL 15 — Model 4: Hybrid Voting Classifier (Proposed Method)

```python
# ============================================================
# CELL 15 — Model 4: Hybrid Voting Classifier (Proposed Method)
# ============================================================

if is_done("hybrid_trained"):
    print("✅ Hybrid model already trained — loading from Drive.")
    hybrid_model   = load_from_drive("models/hybrid_model.pkl")
    hybrid_metrics = load_from_drive("metrics/hybrid_metrics.json")
else:
    if 'lr_model' not in dir() or lr_model is None:
        lr_model = load_from_drive("models/lr_model.pkl")
    if 'rf_model' not in dir() or rf_model is None:
        rf_model = load_from_drive("models/rf_model.pkl")
    if 'xgb_model' not in dir() or xgb_model is None:
        xgb_model = load_from_drive("models/xgb_model.pkl")

    hybrid_model = VotingClassifier(
        estimators=[
            ('lr',  lr_model),
            ('rf',  rf_model),
            ('xgb', xgb_model),
        ],
        voting='soft',
        n_jobs=-1
    )

    X1_train_sm_arr = X1_train_sm.values if hasattr(X1_train_sm, 'values') else X1_train_sm
    X1_test_arr     = X1_test.values     if hasattr(X1_test,     'values') else X1_test

    t0 = time.time()
    hybrid_model.fit(X1_train_sm_arr, y1_train_sm)
    hybrid_train_time = time.time() - t0

    t1 = time.time()
    y_pred_hybrid = hybrid_model.predict(X1_test_arr)
    hybrid_infer_time = (time.time() - t1) / len(X1_test_arr) * 1000

    y_prob_hybrid = hybrid_model.predict_proba(X1_test_arr)[:, 1]

    hybrid_metrics = {
        "accuracy":   accuracy_score(y1_test, y_pred_hybrid),
        "precision":  precision_score(y1_test, y_pred_hybrid, zero_division=0),
        "recall":     recall_score(y1_test, y_pred_hybrid, zero_division=0),
        "f1":         f1_score(y1_test, y_pred_hybrid, zero_division=0),
        "roc_auc":    roc_auc_score(y1_test, y_prob_hybrid),
        "train_time": round(hybrid_train_time, 4),
        "infer_time_ms": round(hybrid_infer_time, 6),
    }

    print("\n📊 Classification Report (Hybrid Voting Classifier):")
    print(classification_report(y1_test, y_pred_hybrid))

    save_to_drive(hybrid_model,   "models/hybrid_model.pkl")
    save_to_drive(hybrid_metrics, "metrics/hybrid_metrics.json")

    mark_done("hybrid_trained")
    print("\n✅ Hybrid model saved — this is the proposed method.")
```

> ### 🧠 Cell 15 Explanation
>
> This cell builds the **Hybrid Soft Voting Classifier — the "proposed method" of this research paper.** This is the main contribution: combining LR + RF + XGBoost into one ensemble model that is more robust and accurate than any single model.
>
> **What is a Voting Classifier?** It's a "committee" model that wraps multiple individually trained models. When given a new URL to classify, it asks each model for its answer, then combines the answers for a final decision.
>
> **Prerequisite check** — Before building the hybrid, the cell verifies all three base models (LR, RF, XGBoost) are loaded in memory. If any are missing (e.g., after a session restart), it loads them from Drive automatically.
>
> **`VotingClassifier(estimators=[...], voting='soft', n_jobs=-1)`**
> - `estimators` — A list of `(name, model)` tuples specifying which models to combine. The names ('lr', 'rf', 'xgb') are just labels used internally.
> - `voting='soft'` — Instead of each model casting a hard vote (0 or 1), each model outputs its probability (e.g., LR says 72% phishing, RF says 88%, XGB says 91%). The final prediction is based on the **average probability**. Soft voting is generally more accurate than hard voting because it uses richer information.
> - `n_jobs=-1` — Parallelizes predictions across CPU cores.
>
> **`hybrid_model.fit(X1_train_sm_arr, y1_train_sm)`** — Even though the three base models were already trained individually, the `VotingClassifier` re-trains all of them internally (using the same data) to ensure they are all aligned to the same training set. This is a scikit-learn design requirement.
>
> Training, prediction, probability estimation, and metrics work the same as Cells 12–14.

---

## CELL 16 — 5-Fold Cross-Validation

```python
# ============================================================
# CELL 16 — 5-Fold Cross-Validation
# ============================================================

if is_done("cv_done"):
    print("✅ CV already done — loading results from Drive.")
    cv_results = load_from_drive("metrics/cv_results.json")
    summary = pd.DataFrame({
        "Model": list(cv_results["mean_f1"].keys()),
        "Mean F1": list(cv_results["mean_f1"].values()),
        "Std F1":  list(cv_results["std_f1"].values()),
    })
    print(summary.to_string(index=False))
else:
    for mname, mkey, mpath in [
        ("lr_model",     "lr_trained",     "models/lr_model.pkl"),
        ("rf_model",     "rf_trained",     "models/rf_model.pkl"),
        ("xgb_model",    "xgb_trained",    "models/xgb_model.pkl"),
        ("hybrid_model", "hybrid_trained", "models/hybrid_model.pkl"),
    ]:
        if mname not in dir() or eval(mname) is None:
            globals()[mname] = load_from_drive(mpath)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    X_all = np.vstack([
        X1_train_sm.values if hasattr(X1_train_sm, 'values') else X1_train_sm,
        X1_test.values     if hasattr(X1_test,     'values') else X1_test
    ])
    y_all = np.concatenate([y1_train_sm, y1_test])

    model_map = {
        "Logistic Regression": lr_model,
        "Random Forest":       rf_model,
        "XGBoost":             xgb_model,
        "Hybrid (Proposed)":   hybrid_model,
    }

    cv_results = {"per_fold": {}, "mean_f1": {}, "std_f1": {}}

    for name, model in model_map.items():
        scores = cross_val_score(model, X_all, y_all,
                                 cv=skf, scoring='f1_weighted', n_jobs=-1)
        cv_results["per_fold"][name]  = scores.tolist()
        cv_results["mean_f1"][name]   = round(scores.mean(), 4)
        cv_results["std_f1"][name]    = round(scores.std(), 4)

    save_to_drive(cv_results, "metrics/cv_results.json")
    mark_done("cv_done")
    print("\n✅ CV results saved to Drive.")
```

> ### 🧠 Cell 16 Explanation
>
> This cell performs **5-Fold Cross-Validation**, a rigorous technique to estimate how well each model will generalize to new, unseen data. It's a scientific way to get a more trustworthy performance estimate than a single train/test split.
>
> **Why cross-validation?** A single 80/20 train/test split might get lucky or unlucky depending on which data ended up in the test set. Cross-validation repeats the evaluation 5 times with different splits, giving a more reliable average.
>
> **`StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)`**
> - Splits the data into 5 equal "folds." In each of the 5 rounds, 4 folds are used for training and 1 fold is used for testing.
> - `shuffle=True` — Randomly shuffles data before splitting (important to avoid ordered biases).
> - `stratify` (implied by `StratifiedKFold`) — Ensures each fold has the same phishing-to-legit ratio as the full dataset.
>
> **`np.vstack([X1_train_sm..., X1_test...])`** — Stacks the training and test arrays vertically (on top of each other) to reassemble the full dataset. Cross-validation will then re-split this combined data itself.
>
> **`np.concatenate([y1_train_sm, y1_test])`** — Concatenates (joins end-to-end) the training and test labels to match the combined feature array.
>
> **`cross_val_score(model, X_all, y_all, cv=skf, scoring='f1_weighted', n_jobs=-1)`**
> - Automatically runs the model through all 5 folds: train on 4, test on 1, repeat.
> - `scoring='f1_weighted'` — Uses weighted F1 as the evaluation metric. "Weighted" means the F1 for each class is weighted by how many examples that class has, giving a balanced view.
> - Returns an array of 5 F1 scores (one per fold).
>
> **`scores.mean()` and `scores.std()`** — The mean gives the average performance; the standard deviation measures consistency. A low std means the model performs reliably across different data splits. Results are saved as `"Mean F1 ± Std F1"` — a common way to report results in research papers.

---

## CELL 17 — Results Consolidation & Comparison Table

```python
# ============================================================
# CELL 17 — Results Consolidation & Comparison Table
# ============================================================

if is_done("results_consolidated"):
    print("✅ Results table already built — loading from Drive.")
    results_df = pd.read_csv(f"{RESULTS_DIR}/comparison_table.csv")
    display(results_df.style.highlight_max(color='lightgreen', axis=0,
                                            subset=results_df.columns[1:]))
else:
    metrics_map = {
        "Logistic Regression": ("lr_metrics",     "metrics/lr_metrics.json"),
        "Random Forest":       ("rf_metrics",     "metrics/rf_metrics.json"),
        "XGBoost":             ("xgb_metrics",    "metrics/xgb_metrics.json"),
        "Hybrid (Proposed)":   ("hybrid_metrics", "metrics/hybrid_metrics.json"),
    }
    loaded_metrics = {}
    for model_name, (var_name, rel_path) in metrics_map.items():
        if var_name in dir() and eval(var_name) is not None:
            loaded_metrics[model_name] = eval(var_name)
        else:
            loaded_metrics[model_name] = load_from_drive(rel_path)

    rows = []
    for model_name, m in loaded_metrics.items():
        if m:
            rows.append({
                "Model":              model_name,
                "Accuracy":           round(m["accuracy"],   4),
                "Precision":          round(m["precision"],  4),
                "Recall":             round(m["recall"],     4),
                "F1-Score":           round(m["f1"],         4),
                "ROC-AUC":            round(m["roc_auc"],    4),
                "Train Time (s)":     round(m["train_time"], 2),
                "Infer Time (ms/s)":  round(m["infer_time_ms"], 4),
            })

    results_df = pd.DataFrame(rows)
    styled = results_df.style.highlight_max(
        color='lightgreen', axis=0,
        subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ).highlight_min(
        color='#ffcccc', axis=0,
        subset=["Train Time (s)", "Infer Time (ms/s)"]
    )
    display(styled)

    if is_done("cv_done"):
        cv_data = load_from_drive("metrics/cv_results.json")
        cv_summary = pd.DataFrame({
            "Model":   list(cv_data["mean_f1"].keys()),
            "CV Mean F1": list(cv_data["mean_f1"].values()),
            "CV Std F1":  list(cv_data["std_f1"].values()),
        })
        display(cv_summary)

    results_df.to_csv(f"{RESULTS_DIR}/comparison_table.csv", index=False)
    styled.to_html(f"{RESULTS_DIR}/comparison_table.html")

    mark_done("results_consolidated")
    print("\n✅ Results consolidated and saved to Drive.")
```

> ### 🧠 Cell 17 Explanation
>
> This cell **collects all four models' performance metrics into a single comparison table** — the central results table of the research paper.
>
> **Loading metrics** — For each model, the cell first checks if the metrics dictionary is already in memory (from the training cell in this session). If not, it loads them from the saved `.json` files on Drive. This allows the cell to work even after a session restart.
>
> **`pd.DataFrame(rows)`** — Converts a list of dictionaries into a DataFrame. Each dictionary is one model's row, and the keys become column headers. The result is a clean table with one row per model.
>
> **`results_df.style.highlight_max(color='lightgreen', axis=0, subset=[...])`**
> - `.style` — Accesses pandas' styling interface for formatting tables for display.
> - `highlight_max(color='lightgreen', axis=0)` — Highlights the cell with the highest value in each column with a green background. `axis=0` means "compare values down each column."
> - `highlight_min(color='#ffcccc', ...)` — Highlights the cell with the lowest training/inference time with a light red background (smaller = faster = better for time).
>
> **`display(styled)`** — Renders the color-highlighted HTML table directly in the notebook output.
>
> **`results_df.to_csv(path, index=False)`** — Saves the table to a CSV file. `index=False` prevents writing the row numbers (0, 1, 2...) as an extra column.
>
> **`styled.to_html(path)`** — Saves the color-highlighted version as an HTML file (a webpage) so it can be opened in a browser and looks exactly as it did in the notebook.

---

## CELL 18 — Publication-Ready Visualizations

```python
# ============================================================
# CELL 18 — Publication-Ready Visualizations (7 plots)
# ============================================================

from IPython.display import Image as IPyImage, display

if is_done("plots_done"):
    print("✅ All plots already done — displaying from Drive.")
    for fname in ["accuracy_comparison.png", "multi_metric_comparison.png",
                  "roc_curves.png", "confusion_matrices.png",
                  "time_vs_accuracy.png", "cv_f1_boxplot.png",
                  "feature_importance_rf.png"]:
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            display(IPyImage(filename=fpath))
else:
    if 'results_df' not in dir():
        results_df = pd.read_csv(f"{RESULTS_DIR}/comparison_table.csv")

    MODEL_COLORS = {
        "Logistic Regression": "#4878CF",
        "Random Forest":       "#6ACC65",
        "XGBoost":             "#D65F5F",
        "Hybrid (Proposed)":   "#B47CC7",
    }

    # ── Plot 1: Accuracy Bar Chart ───────────────────────────
    # ── Plot 2: Multi-Metric Grouped Bar Chart ───────────────
    # ── Plot 3: ROC Curves ───────────────────────────────────
    # ── Plot 4: Confusion Matrices (2×2 grid) ────────────────
    # ── Plot 5: Training Time vs Accuracy Scatter ────────────
    # ── Plot 6: CV F1 Box Plot ───────────────────────────────
    # ── Plot 7: RF Feature Importance (top 20) ───────────────

    mark_done("plots_done")
    print("\n✅ All 7 publication-ready plots saved to Drive.")
```

> ### 🧠 Cell 18 Explanation
>
> This cell **generates all 7 publication-quality plots** comparing the four models. These are the figures that go directly into the research paper.
>
> **`MODEL_COLORS`** — A dictionary assigning a specific hex color code to each model so all 7 plots use a consistent color scheme. E.g., XGBoost is always red (`#D65F5F`), Hybrid is always purple.
>
> **Plot 1 — Accuracy Bar Chart:**
> - A simple bar chart with one bar per model showing its accuracy. `ax.bar(...)` draws the bars. Each bar is labeled with its exact accuracy value using `ax.text(...)`.
>
> **Plot 2 — Multi-Metric Grouped Bar Chart:**
> - Shows Accuracy, Precision, Recall, F1, and ROC-AUC side-by-side for all 4 models. `x + i * width` positions each metric's group of bars without overlap.
>
> **Plot 3 — ROC Curves:**
> - `roc_curve(y1_test, proba)` — Computes the True Positive Rate (TPR) and False Positive Rate (FPR) at many different decision thresholds. Plotting TPR vs FPR gives the ROC curve.
> - A perfect model has a curve hugging the top-left corner (AUC = 1.0). A random model follows the diagonal `k--` dashed line (AUC = 0.5).
> - All 4 models are plotted on the same axes for comparison.
>
> **Plot 4 — Confusion Matrices (2×2 grid):**
> - `confusion_matrix(y1_test, y_pred)` — Returns a 2x2 matrix: [[True Negatives, False Positives], [False Negatives, True Positives]].
> - `sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ...)` — Draws the matrix as a color grid. `annot=True` writes the numbers inside each cell. `fmt='d'` formats numbers as integers. `cmap='Blues'` uses a blue color scale.
> - `plt.subplots(2, 2)` creates a 2-row, 2-column grid of subplots for all 4 models.
>
> **Plot 5 — Training Time vs Accuracy Scatter Plot:**
> - Each model appears as a dot, with training time on the x-axis and accuracy on the y-axis. Models in the top-left are ideal: fast to train AND accurate.
> - `np.polyfit(xs, ys, 1)` fits a 1st-degree polynomial (straight line) through the points. `np.polyval(c, x_line)` evaluates that line across a range of x values for plotting.
>
> **Plot 6 — CV F1 Box Plot:**
> - Shows the distribution of F1 scores across all 5 cross-validation folds for each model. `ax.boxplot(bp_data, ...)` draws boxes where the box spans the interquartile range (Q1 to Q3), the line inside is the median, and whiskers show the range.
>
> **Plot 7 — RF Feature Importance (Top 20):**
> - A horizontal bar chart showing the 20 most important features as determined by the Random Forest.
> - `plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top20)))` — Creates a gradient of colors from red (less important) to green (more important) using a colormap. `np.linspace` generates evenly spaced values.

---

## CELL 19 — SHAP Explainability

```python
# ============================================================
# CELL 19 — SHAP Explainability
# ============================================================

from IPython.display import Image as IPyImage, display

if is_done("shap_done"):
    print("✅ SHAP already done — displaying from Drive.")
    for fname in ["shap_summary.png", "shap_bar.png", "shap_waterfall_example.png"]:
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            display(IPyImage(filename=fpath))
else:
    if xgb_model is None:
        xgb_model = load_from_drive("models/xgb_model.pkl")

    X_test_arr = X1_test.values if hasattr(X1_test, 'values') else X1_test
    feature_names = list(X1_test.columns) if hasattr(X1_test, 'columns') else None

    idx_sample = np.random.choice(len(X_test_arr), size=min(500, len(X_test_arr)), replace=False)
    X_shap = X_test_arr[idx_sample]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_shap)

    # ── Plot 1: SHAP Summary Beeswarm ────────────────────────
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                      plot_type='dot', show=False, max_display=20)
    plt.savefig(f"{PLOTS_DIR}/shap_summary.png", dpi=300, bbox_inches='tight')

    # ── Plot 2: SHAP Bar Plot ─────────────────────────────────
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                      plot_type='bar', show=False, max_display=20)
    plt.savefig(f"{PLOTS_DIR}/shap_bar.png", dpi=300, bbox_inches='tight')

    # ── Plot 3: SHAP Waterfall (single phishing example) ─────
    y_test_arr = np.array(y1_test)
    phishing_idx = np.where(y_test_arr[idx_sample] == 1)[0]
    single_idx = phishing_idx[0] if len(phishing_idx) > 0 else 0

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[single_idx],
            base_values=explainer.expected_value,
            data=X_shap[single_idx],
            feature_names=feature_names,
        ),
        show=False
    )
    plt.savefig(f"{PLOTS_DIR}/shap_waterfall_example.png", dpi=300, bbox_inches='tight')

    mark_done("shap_done")
    print("\n✅ SHAP explainability complete. All 3 plots saved to Drive.")
```

> ### 🧠 Cell 19 Explanation
>
> This cell adds **explainability** to the XGBoost model using SHAP — answering not just "what did the model predict?" but "WHY did it predict that?" This is critical for a research paper and for real-world trust in the model.
>
> **What is SHAP?** SHAP (SHapley Additive exPlanations) is based on game theory. It assigns each feature a "contribution score" (SHAP value) for a specific prediction: positive values push toward "phishing," negative values push toward "legitimate."
>
> **`np.random.choice(len(X_test_arr), size=500, replace=False)`** — Randomly selects 500 unique indices from the test set. Computing SHAP values for all test samples would be very slow, so a 500-sample subset is used. `replace=False` ensures no sample is selected twice.
>
> **`shap.TreeExplainer(xgb_model)`** — Creates a SHAP explainer specifically optimized for tree-based models (like XGBoost, Random Forest). It uses the internal structure of the trees to calculate SHAP values exactly and very efficiently.
>
> **`explainer.shap_values(X_shap)`** — Computes SHAP values for all 500 samples. The result is a matrix with the same shape as `X_shap` — one SHAP value per feature per sample.
>
> **Plot 1 — Beeswarm (Summary) Plot:**
> - `shap.summary_plot(..., plot_type='dot')` — For each feature (y-axis), each dot represents one sample. Dot position on the x-axis = that sample's SHAP value for this feature. Dot color = the actual feature value (red = high, blue = low). This shows both the magnitude and direction of each feature's influence across all samples.
>
> **Plot 2 — SHAP Bar Plot:**
> - `plot_type='bar'` — Shows the mean absolute SHAP value for each feature — a simple overall ranking of feature importance.
>
> **Plot 3 — Waterfall Plot (single URL explanation):**
> - `np.where(y_test_arr[idx_sample] == 1)[0]` — Finds indices within the 500-sample subset where the true label is phishing (class 1). `[0]` picks the first such index.
> - `shap.Explanation(values=..., base_values=..., data=..., feature_names=...)` — Bundles the SHAP values, the model's baseline prediction (average prediction across all data), the actual feature values, and feature names into an Explanation object.
> - `shap.waterfall_plot(...)` — Shows a single URL's prediction as a waterfall: starting from the baseline prediction, each feature's SHAP value either adds to or subtracts from the probability, ending at the final prediction. This is a precise, human-readable explanation of one specific decision.

---

## CELL 20 — Cross-Dataset Validation

```python
# ============================================================
# CELL 20 — Cross-Dataset Validation
# ============================================================

if is_done("cross_dataset_done"):
    print("✅ Cross-dataset validation already done.")
    cross_results = load_from_drive("metrics/cross_dataset_results.json")
    print(pd.DataFrame(cross_results).to_string())
else:
    if 'X2_test' not in dir():
        X2_train = load_from_drive("checkpoints/preprocessed_X2_train.pkl")
        X2_test  = load_from_drive("checkpoints/preprocessed_X2_test.pkl")
        y2_train = load_from_drive("checkpoints/preprocessed_y2_train.pkl")
        y2_test  = load_from_drive("checkpoints/preprocessed_y2_test.pkl")

    X1_cols = set(X1_train.columns) if hasattr(X1_train, 'columns') else set()
    X2_cols = set(X2_test.columns)  if hasattr(X2_test,  'columns') else set()
    common  = X1_cols & X2_cols
    print(f"📐 Feature overlap DS1 ∩ DS2: {len(common)} common features")

    cross_results = {"method": [], "model": [], "accuracy": [], "f1": [], "roc_auc": []}

    if len(common) >= 10:
        common_list = sorted(common)
        X2_sub = X2_test[common_list]
        X2_sub_arr = X2_sub.values

        for model_name, model in [("Random Forest", rf_model), ("Hybrid (Proposed)", hybrid_model)]:
            try:
                preds = model.predict(X2_sub_arr)
                proba = model.predict_proba(X2_sub_arr)[:, 1]
                cross_results["method"].append("common_features")
                cross_results["model"].append(model_name)
                cross_results["accuracy"].append(round(accuracy_score(y2_test, preds), 4))
                cross_results["f1"].append(round(f1_score(y2_test, preds, zero_division=0), 4))
                cross_results["roc_auc"].append(round(roc_auc_score(y2_test, proba), 4))
            except Exception as e:
                print(f"  ⚠️  {model_name} failed: {e}")
    else:
        from sklearn.ensemble import RandomForestClassifier as _RFC
        _rf2 = _RFC(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
        _rf2.fit(X2_train_arr, y2_train)
        _lr2 = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, n_jobs=-1)
        _lr2.fit(X2_train_arr, y2_train)
        _hybrid2 = VotingClassifier(estimators=[('lr', _lr2), ('rf', _rf2)], voting='soft', n_jobs=-1)
        _hybrid2.fit(X2_train_arr, y2_train)
        for model_name, model in [("RF (fresh DS2)", _rf2), ("Hybrid (fresh DS2)", _hybrid2)]:
            preds = model.predict(X2_test_arr)
            proba = model.predict_proba(X2_test_arr)[:, 1]
            cross_results["method"].append("fresh_ds2")
            cross_results["model"].append(model_name)
            cross_results["accuracy"].append(round(accuracy_score(y2_test, preds), 4))
            cross_results["f1"].append(round(f1_score(y2_test, preds, zero_division=0), 4))
            cross_results["roc_auc"].append(round(roc_auc_score(y2_test, proba), 4))

    save_to_drive(cross_results, "metrics/cross_dataset_results.json")
    mark_done("cross_dataset_done")
    print(pd.DataFrame(cross_results).to_string(index=False))
    print("\n✅ Cross-dataset results saved to Drive.")
```

> ### 🧠 Cell 20 Explanation
>
> This cell tests whether the models trained on Dataset 1 can also work well on Dataset 2 — a completely different collection of phishing data. This is called **cross-dataset generalization** and is a key indicator of whether the model has learned real, universal patterns of phishing vs. just memorized the quirks of one dataset.
>
> **`X1_cols & X2_cols`** — The `&` operator on Python sets gives the **intersection**: only column names that appear in both datasets. These are the only features that both models can use for comparison.
>
> **Two paths based on feature overlap:**
>
> **If ≥ 10 common features exist:**
> - `X2_test[common_list]` — Takes only the common columns from the Dataset 2 test set, so the DS1-trained models can score them.
> - Models trained on DS1 are tested directly on DS2 data (using shared features). If accuracy is still good, it proves the model learned generalizable patterns.
> - Wrapped in `try/except` because the feature dimensions may mismatch or the scaler may not align — the exception handler catches these gracefully.
>
> **If fewer than 10 common features:**
> - The datasets are too different to directly transfer. Fresh models are trained on DS2's own training data and tested on DS2's test data. This still validates the same **methodology** works on a second dataset, even if direct transfer isn't possible.
>
> **`pd.DataFrame(cross_results).to_string(index=False)`** — Converts the results dictionary to a printable table without row numbers.

---

## CELL 21 — Lightweight Inference Demo

```python
# ============================================================
# CELL 21 — Lightweight Inference Demo
# ============================================================

from IPython.display import Image as IPyImage, display

if is_done("inference_demo_done"):
    print("✅ Inference demo already done.")
    display(IPyImage(filename=f"{PLOTS_DIR}/inference_latency.png"))
else:
    X_test_arr = X1_test.values if hasattr(X1_test, 'values') else X1_test

    for mname, mpath in [("lr_model", "models/lr_model.pkl"), ...]:
        if mname not in dir() or globals().get(mname) is None:
            globals()[mname] = load_from_drive(mpath)

    # ── Part A: Batch Latency Benchmark ──────────────────────
    batch_sizes = [1000, 5000, 10000]
    model_map   = {"Logistic Regression": lr_model, "Random Forest": rf_model,
                   "XGBoost": xgb_model, "Hybrid (Proposed)": hybrid_model}
    latency_data = {name: [] for name in model_map}

    for n in batch_sizes:
        X_batch = X_test_arr[:n]
        for model_name, model in model_map.items():
            if model is not None:
                t0 = time.time()
                _ = model.predict(X_batch)
                elapsed_ms = (time.time() - t0) * 1000
                latency_data[model_name].append(elapsed_ms)

    # Plot latency line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, latencies in latency_data.items():
        ax.plot(batch_sizes[:len(latencies)], latencies, marker='o', linewidth=2,
                color=MODEL_COLORS.get(model_name, "#888"), label=model_name)
    ax.set_xlabel("Batch Size (samples)"); ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Inference Latency by Batch Size", fontsize=14, fontweight='bold')
    ax.legend(); plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/inference_latency.png", dpi=300)

    # ── Part B: Single URL Demo ───────────────────────────────
    fake_url_features = pd.DataFrame(np.zeros((1, X_test_arr.shape[1])), columns=X1_test.columns)
    for col in fake_url_features.columns:
        if 'length' in col or 'count' in col:
            fake_url_features[col] = 150.0
    pred_class = hybrid_model.predict(fake_url_features.values)[0]
    pred_proba = hybrid_model.predict_proba(fake_url_features.values)[0]
    confidence = pred_proba[pred_class] * 100
    label_str  = "PHISHING" if pred_class == 1 else "LEGITIMATE"

    # ── Part C: Model File Size Table ────────────────────────
    for model_name, model in model_map.items():
        fpath = os.path.join(tmp_dir, f"{model_name.replace(' ', '_')}.pkl")
        joblib.dump(model, fpath)
        size_kb = os.path.getsize(fpath) / 1024

    mark_done("inference_demo_done")
    print("\n✅ Inference demo complete.")
```

> ### 🧠 Cell 21 Explanation
>
> This cell demonstrates **how fast and practical the models are in real deployment scenarios** — a key requirement for a "lightweight" system as described in the paper title.
>
> **Part A — Batch Latency Benchmark:**
> - Tests each model's prediction speed on batches of 1,000, 5,000, and 10,000 URLs.
> - `X_test_arr[:n]` — Slices the first `n` rows of the test set as the batch.
> - `(time.time() - t0) * 1000` — Measures total prediction time in milliseconds (×1000 to convert from seconds).
> - The line chart shows how each model's inference time grows with batch size. A flatter line = more scalable model.
>
> **Part B — Single URL Demo:**
> - `np.zeros((1, X_test_arr.shape[1]))` — Creates a single feature vector of all zeros, representing a "fake" URL.
> - Features with "length" or "count" in their name are set to 150 — simulating a suspiciously long URL (a common phishing indicator).
> - The hybrid model predicts whether this fake URL is phishing or legitimate and prints the confidence percentage.
> - `pred_proba[pred_class]` — Gets the probability for the predicted class. Multiplying by 100 converts 0.91 → 91%.
>
> **Part C — Model File Size Table:**
> - `joblib.dump(model, fpath)` — Saves each model as a `.pkl` file.
> - `os.path.getsize(fpath) / 1024` — Gets the file size in bytes and divides by 1024 to convert to kilobytes (KB).
> - This tells you how much disk/memory each model requires for deployment — important for the "lightweight" claim in the paper.

---

## CELL 22 — Paper-Ready Summary Report

```python
# ============================================================
# CELL 22 — Paper-Ready Summary Report
# ============================================================

if not is_done("results_consolidated"):
    print("❌ ERROR: results_consolidated not done. Run Cell 17 first.")
else:
    if 'results_df' not in dir():
        results_df = pd.read_csv(f"{RESULTS_DIR}/comparison_table.csv")

    cv_data    = load_from_drive("metrics/cv_results.json") if is_done("cv_done") else {}
    cross_data = load_from_drive("metrics/cross_dataset_results.json") \
                 if is_done("cross_dataset_done") else {}

    state = load_checkpoint()
    best_model_row = results_df.loc[results_df["Accuracy"].idxmax()]
    best_model     = best_model_row["Model"]

    report_lines = [
        "=" * 70,
        "  PHISHING DETECTION — PAPER-READY SUMMARY REPORT",
        ...
    ]

    # Add model comparison table rows, CV results, cross-dataset results,
    # plot artifact list, and research paper section mapping

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = f"{RESULTS_DIR}/summary_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\n📄 Summary report saved to Drive.")
```

> ### 🧠 Cell 22 Explanation
>
> This is the **final cell** — it generates a complete, formatted text report that summarizes the entire experiment. This is designed to be directly usable as a draft for the research paper's results section.
>
> **`results_df["Accuracy"].idxmax()`** — `idxmax()` returns the **index (row number)** of the row with the maximum value in the "Accuracy" column. `results_df.loc[...]` then retrieves that entire row. This identifies which model had the highest accuracy.
>
> **`report_lines`** — A Python list of strings. Each string is one line of the final report. This approach makes it easy to build a long text document dynamically by appending lines.
>
> **Report contents:**
> - Dataset summary (shapes, target columns — loaded from the checkpoint's metadata).
> - Full model performance comparison table (formatted with fixed-width columns using f-strings like `f"{str(row['Model']):<28}"` where `<28` means "left-align in a 28-character field").
> - Best model highlighted with its key metrics.
> - Cross-validation results (Mean F1 ± Std F1 per model).
> - Cross-dataset validation results.
> - List of all saved plot files.
> - **Research Paper Mapping** — a section showing which notebook cell corresponds to which section of the paper (Abstract, Introduction, Methodology, Results, etc.). This is a convenience guide for writing the paper.
>
> **`"\n".join(report_lines)`** — Combines all the strings in `report_lines` into a single multi-line string, with each original string on its own line.
>
> **`with open(report_path, "w") as f: f.write(report_text)`** — Opens a file for writing (`"w"` mode) and writes the complete report text to it. The `with` statement ensures the file is properly closed after writing, even if an error occurs.
>
> The report is both printed in the notebook and saved as `summary_report.txt` on Google Drive for later reference.