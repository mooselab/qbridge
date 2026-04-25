import os
import json
import warnings
import joblib

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

DATA_ROOT = "./data/qlear_pretrain_by_backend"
MODEL_ROOT = "./model/qlear_pretrain_models_keras"

RANDOM_STATE = 42
TRAIN_SIZE = 0.8
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


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def list_backend_csvs(data_root):
    return sorted(
        os.path.join(data_root, f)
        for f in os.listdir(data_root)
        if os.path.isfile(os.path.join(data_root, f)) and f.endswith(".csv")
    )


def load_dataframe(csv_path):
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


def build_qlear_mlp(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,), name="x")
    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.Dense(1000, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="qlear_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LR),
        loss="mse",
        metrics=["mse"],
    )
    return model


def train_one_backend(csv_path):
    backend_name = os.path.basename(csv_path).replace(".csv", "")
    print(f"\n=== Training backend: {backend_name} ===")

    df = load_dataframe(csv_path)
    if len(df) == 0:
        raise ValueError(f"No valid rows after cleaning for {csv_path}")

    train_df, val_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        test_size=1 - TRAIN_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    x_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df[LABEL].to_numpy(dtype=np.float32).reshape(-1, 1)
    x_val = val_df[FEATURES].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL].to_numpy(dtype=np.float32).reshape(-1, 1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    model = build_qlear_mlp(input_dim=len(FEATURES))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )

    backend_dir = os.path.join(MODEL_ROOT, backend_name)
    os.makedirs(backend_dir, exist_ok=True)

    model.save(os.path.join(backend_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(backend_dir, "scaler.pkl"))

    meta = {
        "backend": backend_name,
        "features": FEATURES,
        "label": LABEL,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "model_format": "keras",
        "uses_embedding": False,
    }
    with open(os.path.join(backend_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {backend_dir}")
    return meta


def main():
    set_seed(RANDOM_STATE)
    os.makedirs(MODEL_ROOT, exist_ok=True)

    files = list_backend_csvs(DATA_ROOT)
    if not files:
        raise FileNotFoundError(f"No backend csv files found under: {DATA_ROOT}")

    summaries = []
    for csv_path in tqdm(files, desc="Pretrain backends"):
        try:
            meta = train_one_backend(csv_path)
            summaries.append(meta)
        except Exception as ex:
            print(f"[Skip] Failed on {csv_path}: {ex}")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            os.path.join(MODEL_ROOT, "all_backend_training_summary.csv"),
            index=False
        )


if __name__ == "__main__":
    main()
