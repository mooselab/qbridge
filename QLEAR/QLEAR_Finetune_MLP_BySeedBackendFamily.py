import os
import json
import random
import warnings
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

DATA_ROOT = "./data/qlear_finetune_by_backend_family"
PRETRAIN_MODEL_ROOT = "./model/qlear_pretrain_models_keras"
MODEL_ROOT = "./model/qlear_finetune_models_keras"

SEEDS = [1, 2, 3]
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 10

FEATURES = [
    "Avg_inverted_error_25",
    "Avg_inverted_error_50",
    "Avg_inverted_error_75",
    "Avg_odds_ratio",
    "Num_1Q_Gates",
    "Num_2Q_Gates",
    "circuit_depth",
    "circuit_width",
    "observed_prob_50",
    "state_weight",
]
LABEL = "target"


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def list_backend_dirs(data_root):
    return sorted(
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    )


def list_family_csvs(backend_dir):
    return sorted(
        os.path.join(backend_dir, f)
        for f in os.listdir(backend_dir)
        if os.path.isfile(os.path.join(backend_dir, f)) and f.endswith(".csv")
    )


def parse_backend_name_from_dir(backend_dir):
    return os.path.basename(backend_dir)


def parse_family_name_from_csv(csv_path, backend_name):
    stem = os.path.basename(csv_path).replace(".csv", "")
    prefix = f"{backend_name}_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def load_family_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    required = FEATURES + [LABEL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    df = df.loc[:, required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


def train_one_seed_backend_family(seed, backend_name, csv_path):
    family_name = parse_family_name_from_csv(csv_path, backend_name)
    pretrain_dir = os.path.join(PRETRAIN_MODEL_ROOT, backend_name)

    model_path = os.path.join(pretrain_dir, "model.keras")
    scaler_path = os.path.join(pretrain_dir, "scaler.pkl")
    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Missing keras pretrain artifacts in: {pretrain_dir}")

    set_seed(seed)

    df = load_family_dataframe(csv_path)
    if len(df) == 0:
        raise ValueError(f"No valid rows after cleaning for {csv_path}")

    x = df[FEATURES].to_numpy(dtype=np.float32)
    y = df[LABEL].to_numpy(dtype=np.float32).reshape(-1, 1)

    scaler = joblib.load(scaler_path)
    x = scaler.transform(x).astype(np.float32)

    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR),
        loss="mse",
        metrics=["mse"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        x,
        y,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )

    save_dir = os.path.join(MODEL_ROOT, f"seed_{seed}", f"{backend_name}_{family_name}")
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    meta = {
        "seed": int(seed),
        "backend": backend_name,
        "family": family_name,
        "features": FEATURES,
        "label": LABEL,
        "rows": int(len(df)),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "uses_embedding": False,
    }
    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {save_dir}")
    return meta


def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)

    backend_dirs = list_backend_dirs(DATA_ROOT)
    if not backend_dirs:
        raise FileNotFoundError(f"No backend directories found under: {DATA_ROOT}")

    summaries = []
    for seed in SEEDS:
        print("\\n" + "=" * 72)
        print(f"Current seed: {seed}")
        print("=" * 72)

        for backend_dir in tqdm(backend_dirs, desc=f"Seed {seed} backends"):
            backend_name = parse_backend_name_from_dir(backend_dir)
            family_csvs = list_family_csvs(backend_dir)

            for csv_path in family_csvs:
                try:
                    meta = train_one_seed_backend_family(seed, backend_name, csv_path)
                    summaries.append(meta)
                except Exception as ex:
                    print(f"[Skip] Failed on seed={seed}, backend={backend_name}, file={csv_path}: {ex}")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            os.path.join(MODEL_ROOT, "all_finetune_training_summary.csv"),
            index=False
        )


if __name__ == "__main__":
    main()
