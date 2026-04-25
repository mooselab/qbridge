import os
import json
import pickle
import random
import warnings
from collections import defaultdict

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

DATA_ROOT = "./data/qlear_test_by_backend_family"
MODEL_ROOT = "./model/qlear_finetune_models_keras"
RESULT_ROOT = "./qlear_test_results_keras"

SEEDS = [1, 2, 3]

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


def hellinger_distance(p, q):
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)

    p[p < 0] = 0.0
    q[q < 0] = 0.0

    sp = p.sum()
    sq = q.sum()

    if sp > 0:
        p = p / sp
    if sq > 0:
        q = q / sq

    return (1.0 / np.sqrt(2.0)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def clip_predictions(pred):
    arr = np.asarray(pred, dtype=float).reshape(-1)
    arr[arr < 0] = 0.0
    arr[arr > 1] = 1.0
    return arr


def list_backend_dirs(data_root: str):
    dirs = []
    for name in os.listdir(data_root):
        full = os.path.join(data_root, name)
        if os.path.isdir(full):
            dirs.append(full)
    return sorted(dirs)


def list_family_csvs(backend_dir: str):
    files = []
    for name in os.listdir(backend_dir):
        full = os.path.join(backend_dir, name)
        if os.path.isfile(full) and name.endswith(".csv"):
            files.append(full)
    return sorted(files)


def parse_backend_name_from_dir(backend_dir: str):
    return os.path.basename(backend_dir)


def parse_family_name_from_csv(csv_path: str, backend_name: str):
    stem = os.path.basename(csv_path).replace(".csv", "")
    prefix = f"{backend_name}_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def load_test_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"circuit": "string"})
    required_cols = FEATURES + [LABEL, "circuit"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    for c in FEATURES + [LABEL]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df["circuit"] = df["circuit"].astype(str)
    return df


def load_finetuned_artifacts(seed: int, backend_name: str, family_name: str):
    model_dir = os.path.join(MODEL_ROOT, f"seed_{seed}", f"{backend_name}_{family_name}")
    model_path = os.path.join(model_dir, "model.keras")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Finetuned model not found: {model_path}")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Finetuned scaler not found: {scaler_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model_dir, model, scaler


def evaluate_one_seed_backend_family(seed: int, backend_name: str, csv_path: str):
    family_name = parse_family_name_from_csv(csv_path, backend_name)
    model_dir, model, scaler = load_finetuned_artifacts(seed, backend_name, family_name)

    set_seed(seed)
    df = load_test_dataframe(csv_path)

    per_circuit_hl = []
    grouped = df.groupby("circuit", sort=True)

    for circuit_id, sub in grouped:
        x = sub[FEATURES].to_numpy(dtype=np.float32)
        y_true = sub[LABEL].to_numpy(dtype=np.float32).reshape(-1)

        x = scaler.transform(x).astype(np.float32)
        pred = model.predict(x, verbose=0).reshape(-1)
        y_pred = clip_predictions(pred)

        hl = hellinger_distance(y_true, y_pred)
        per_circuit_hl.append({
            "seed": seed,
            "backend": backend_name,
            "family": family_name,
            "circuit": circuit_id,
            "n_states": int(len(sub)),
            "hellinger": float(hl),
        })

    per_circuit_df = pd.DataFrame(per_circuit_hl)
    summary = {
        "seed": int(seed),
        "backend": backend_name,
        "family": family_name,
        "n_circuits": int(len(per_circuit_df)),
        "mean_hellinger": float(per_circuit_df["hellinger"].mean()) if len(per_circuit_df) > 0 else None,
        "std_hellinger": float(per_circuit_df["hellinger"].std(ddof=0)) if len(per_circuit_df) > 0 else None,
        "model_dir": model_dir,
        "csv_path": csv_path,
    }
    return per_circuit_df, summary


def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)

    backend_dirs = list_backend_dirs(DATA_ROOT)
    if not backend_dirs:
        raise FileNotFoundError(f"No backend directories found under: {DATA_ROOT}")

    all_per_circuit = []
    all_summaries = []

    for seed in SEEDS:
        print("\n" + "=" * 72)
        print(f"Current seed: {seed}")
        print("=" * 72)

        set_seed(seed)
        seed_dir = os.path.join(RESULT_ROOT, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        nested_results = defaultdict(dict)

        for backend_dir in tqdm(backend_dirs, desc=f"Seed {seed} backends"):
            backend_name = parse_backend_name_from_dir(backend_dir)
            family_csvs = list_family_csvs(backend_dir)

            for csv_path in family_csvs:
                try:
                    per_circuit_df, summary = evaluate_one_seed_backend_family(seed, backend_name, csv_path)
                    family_name = summary["family"]

                    out_csv = os.path.join(seed_dir, f"{backend_name}_{family_name}_per_circuit.csv")
                    per_circuit_df.to_csv(out_csv, index=False)

                    nested_results[backend_name][family_name] = per_circuit_df["hellinger"].tolist()

                    summary_path = os.path.join(seed_dir, f"{backend_name}_{family_name}_summary.json")
                    with open(summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2)

                    all_per_circuit.append(per_circuit_df)
                    all_summaries.append(summary)

                    print(f"Saved: {out_csv}")
                    print(f"Mean Hellinger: {summary['mean_hellinger']}")
                except Exception as ex:
                    print(f"[Skip] Failed on seed={seed}, backend={backend_name}, file={csv_path}: {ex}")

        with open(os.path.join(seed_dir, "saveRQ1.pkl"), "wb") as f:
            pickle.dump({k: dict(v) for k, v in nested_results.items()}, f)

    if all_per_circuit:
        pd.concat(all_per_circuit, ignore_index=True).to_csv(
            os.path.join(RESULT_ROOT, "all_per_circuit_hellinger.csv"), index=False
        )

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(
            os.path.join(RESULT_ROOT, "all_family_mean_hellinger.csv"), index=False
        )


if __name__ == "__main__":
    main()
