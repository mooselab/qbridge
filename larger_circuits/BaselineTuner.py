import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import hashlib
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from qiskit.exceptions import MissingOptionalLibraryError

from benchmark_circuits import *

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

BACKENDS = [
    ('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65),
    ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28),
    ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJohannesburg', 20),
    ('FakeKolkata', 27), ('FakeManhattan', 65), ('FakeMontreal', 27),
    ('FakeMumbai', 27), ('FakeParis', 27), ('FakeRochester', 53),
    ('FakeSingapore', 20), ('FakeSydney', 27), ('FakeToronto', 27),
    ('FakeWashington', 127)
]

SEEDS = [1, 2, 3]
TRAIN_PER_FAMILY = 50
TARGET_EVAL_TOTAL = 60
IDEAL_EVAL_PER_FAMILY = 10

QUBIT_MIN = 8
QUBIT_MAX = 15

BASELINE_TUNING_DIR = "data/data/baseline_tunning_data"
MODEL_DIR = "model/tunning_models"
SPLIT_MANIFEST_PATH = "scaled_family_input_split_manifest.json"
PROGRESS_DIR = "data/baseline_tunning_progress"

MAX_TRAIN_UNIQUE_PER_FAMILY = 80
MAX_EVAL_UNIQUE_PER_FAMILY = 20

FAMILY_FILTER = os.environ.get("FAMILY_FILTER", "").strip()
BACKEND_FILTER = os.environ.get("BACKEND_FILTER", "").strip()
MAX_BACKENDS = int(os.environ.get("MAX_BACKENDS", "0") or "0")


def set_seed(seed):
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def normalize_item(x):
    if isinstance(x, tuple):
        return tuple(normalize_item(v) for v in x)
    if isinstance(x, list):
        return tuple(normalize_item(v) for v in x)
    return x


def serialize_item(x):
    if isinstance(x, tuple):
        return [serialize_item(v) for v in x]
    if isinstance(x, list):
        return [serialize_item(v) for v in x]
    return x


def deserialize_item(x):
    if isinstance(x, list):
        return tuple(deserialize_item(v) for v in x)
    return x


def get_family_split():
    baseline_circuits, cuts = train_test_split(
        get_all_circuits(),
        train_size=0.4,
        random_state=13,
    )
    return baseline_circuits, cuts


def circuit_fingerprint(qc):
    try:
        qasm = qc.qasm()
    except Exception:
        qasm = str(qc)
    return hashlib.sha256(qasm.encode("utf-8")).hexdigest(), qasm


def inspect_input(family_name, inp, backend_ideal):
    circuit = get_circuit_class_object(family_name)
    try:
        circuit.get_result(backend_ideal, inp, use_gpu=True)
        qc = circuit.key_aurguments["circuit"]
        if qc is None:
            return None
        n_qubits = int(qc.num_qubits)
        qhash, qasm = circuit_fingerprint(qc)
        return {
            "input": normalize_item(inp),
            "num_qubits": n_qubits,
            "qasm_hash": qhash,
            "qasm": qasm,
        }
    except Exception as ex:
        print(f"[Skip] Family={family_name} input={inp} failed during inspection: {ex}")
        return None


def collect_unique_candidates(family_name, raw_inputs, backend_ideal, max_keep=None):
    unique_by_hash = {}
    total = len(raw_inputs) if hasattr(raw_inputs, "__len__") else None

    for idx, raw in enumerate(raw_inputs):
        if idx % 10 == 0:
            if total is None:
                print(f"[manifest] {family_name}: inspected {idx}")
            else:
                print(f"[manifest] {family_name}: inspected {idx}/{total}")

        info = inspect_input(family_name, normalize_item(raw), backend_ideal)
        if info is None:
            continue
        if not (QUBIT_MIN <= info["num_qubits"] <= QUBIT_MAX):
            continue
        if info["qasm_hash"] not in unique_by_hash:
            unique_by_hash[info["qasm_hash"]] = info
            if max_keep is not None and len(unique_by_hash) >= max_keep:
                break

    candidates = list(unique_by_hash.values())
    candidates.sort(key=lambda x: (x["num_qubits"], repr(x["input"]), x["qasm_hash"]))
    print(f"[manifest] {family_name}: kept {len(candidates)} unique {QUBIT_MIN}-{QUBIT_MAX}q candidates")
    return candidates


def allocate_eval_counts(eval_pools, target_total=TARGET_EVAL_TOTAL, ideal_per_family=IDEAL_EVAL_PER_FAMILY):
    counts = {family: min(ideal_per_family, len(pool)) for family, pool in eval_pools.items()}
    current_total = sum(counts.values())
    if current_total >= target_total:
        return counts

    capacities = {family: len(pool) - counts[family] for family, pool in eval_pools.items()}
    while current_total < target_total:
        made_progress = False
        for family in sorted(eval_pools.keys()):
            if capacities[family] > 0:
                counts[family] += 1
                capacities[family] -= 1
                current_total += 1
                made_progress = True
                if current_total >= target_total:
                    break
        if not made_progress:
            break
    return counts


def build_split_manifest(backend_ideal):
    baseline_circuits, cuts = get_family_split()
    payload = {
        "BaselineCircuits": baseline_circuits,
        "CUTs": cuts,
        "train_per_family": TRAIN_PER_FAMILY,
        "target_eval_total": TARGET_EVAL_TOTAL,
        "ideal_eval_per_family": IDEAL_EVAL_PER_FAMILY,
        "qubit_min": QUBIT_MIN,
        "qubit_max": QUBIT_MAX,
        "families": {},
    }

    eval_pools = {}

    for family_name in cuts:
        print(f"[manifest] building family {family_name}")
        circuit = get_circuit_class_object(family_name)

        train_candidates = collect_unique_candidates(
            family_name,
            list(circuit.get_inputs()),
            backend_ideal,
            max_keep=MAX_TRAIN_UNIQUE_PER_FAMILY,
        )

        full_candidates = collect_unique_candidates(
            family_name,
            list(circuit.get_full_inputs()),
            backend_ideal,
            max_keep=MAX_EVAL_UNIQUE_PER_FAMILY + max(1, len(train_candidates)),
        )

        if not train_candidates:
            raise RuntimeError(f"No valid {QUBIT_MIN}-{QUBIT_MAX} qubit train candidates for family {family_name}")

        train_hashes = {x["qasm_hash"] for x in train_candidates}
        train_inputs_repr = {repr(x["input"]) for x in train_candidates}

        eval_pool = [
            x for x in full_candidates
            if x["qasm_hash"] not in train_hashes and repr(x["input"]) not in train_inputs_repr
        ]
        eval_pools[family_name] = eval_pool

        payload["families"][family_name] = {
            "input_type": int(circuit.key_aurguments["input_type"]),
            "train_candidates": train_candidates,
            "eval_pool": eval_pool,
        }

    eval_counts = allocate_eval_counts(eval_pools)
    payload["actual_eval_total"] = int(sum(eval_counts.values()))
    payload["eval_counts"] = eval_counts

    for family_name in cuts:
        train_candidates = payload["families"][family_name]["train_candidates"]
        eval_pool = payload["families"][family_name]["eval_pool"]

        train_inputs = [train_candidates[i % len(train_candidates)]["input"] for i in range(TRAIN_PER_FAMILY)]
        eval_inputs = [eval_pool[i]["input"] for i in range(eval_counts[family_name])]

        payload["families"][family_name]["train_inputs"] = train_inputs
        payload["families"][family_name]["eval_inputs"] = eval_inputs
        payload["families"][family_name]["train_qasm_hashes"] = [x["qasm_hash"] for x in train_candidates]
        payload["families"][family_name]["eval_qasm_hashes"] = [x["qasm_hash"] for x in eval_pool[:eval_counts[family_name]]]

        overlap = (
            set(payload["families"][family_name]["train_qasm_hashes"])
            & set(payload["families"][family_name]["eval_qasm_hashes"])
        )
        if overlap:
            raise AssertionError(f"Train/eval QASM overlap detected for family {family_name}")

    return payload


def save_manifest(payload, path=SPLIT_MANIFEST_PATH):
    serializable = {
        "BaselineCircuits": payload["BaselineCircuits"],
        "CUTs": payload["CUTs"],
        "train_per_family": payload["train_per_family"],
        "target_eval_total": payload["target_eval_total"],
        "ideal_eval_per_family": payload["ideal_eval_per_family"],
        "actual_eval_total": payload["actual_eval_total"],
        "eval_counts": payload["eval_counts"],
        "qubit_min": payload["qubit_min"],
        "qubit_max": payload["qubit_max"],
        "families": {},
    }
    for family_name, info in payload["families"].items():
        serializable["families"][family_name] = {
            "input_type": info["input_type"],
            "train_candidates": [
                {
                    "input": serialize_item(x["input"]),
                    "num_qubits": x["num_qubits"],
                    "qasm_hash": x["qasm_hash"],
                }
                for x in info["train_candidates"]
            ],
            "eval_pool": [
                {
                    "input": serialize_item(x["input"]),
                    "num_qubits": x["num_qubits"],
                    "qasm_hash": x["qasm_hash"],
                }
                for x in info["eval_pool"]
            ],
            "train_inputs": [serialize_item(x) for x in info["train_inputs"]],
            "eval_inputs": [serialize_item(x) for x in info["eval_inputs"]],
            "train_qasm_hashes": info["train_qasm_hashes"],
            "eval_qasm_hashes": info["eval_qasm_hashes"],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def manifest_schema_ok(raw):
    required_top = [
        "CUTs", "families", "actual_eval_total", "eval_counts",
        "train_per_family", "target_eval_total", "ideal_eval_per_family",
        "qubit_min", "qubit_max",
    ]
    for key in required_top:
        if key not in raw:
            return False

    if raw["train_per_family"] != TRAIN_PER_FAMILY:
        return False
    if raw["target_eval_total"] != TARGET_EVAL_TOTAL:
        return False
    if raw["ideal_eval_per_family"] != IDEAL_EVAL_PER_FAMILY:
        return False
    if raw["qubit_min"] != QUBIT_MIN:
        return False
    if raw["qubit_max"] != QUBIT_MAX:
        return False

    if not isinstance(raw["families"], dict):
        return False

    for _, info in raw["families"].items():
        for key in ["train_inputs", "eval_inputs", "train_qasm_hashes", "eval_qasm_hashes"]:
            if key not in info:
                return False
    return True


def load_manifest(path=SPLIT_MANIFEST_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not manifest_schema_ok(raw):
        raise ValueError("Manifest schema is outdated or invalid")

    for family_name, info in raw["families"].items():
        info["train_inputs"] = [deserialize_item(x) for x in info["train_inputs"]]
        info["eval_inputs"] = [deserialize_item(x) for x in info["eval_inputs"]]
    return raw


def load_or_build_manifest(backend_ideal, path=SPLIT_MANIFEST_PATH):
    if os.path.exists(path):
        try:
            raw = load_manifest(path)
            print(f"[manifest] loaded existing manifest: {path}")
            return raw
        except Exception as ex:
            print(f"[manifest] existing manifest invalid, rebuilding: {ex}")

    payload = build_split_manifest(backend_ideal)
    save_manifest(payload, path)
    print(f"[manifest] wrote new manifest: {path}")
    return load_manifest(path)


def ensure_dirs():
    os.makedirs(BASELINE_TUNING_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    for bk, _ in BACKENDS:
        os.makedirs(os.path.join(BASELINE_TUNING_DIR, bk), exist_ok=True)
        # os.makedirs(os.path.join("..", "data", BASELINE_TUNING_DIR, bk), exist_ok=True)


def csv_output_path(backend_name, family_name):
    return os.path.join(BASELINE_TUNING_DIR, backend_name, f"{backend_name}_{family_name}.csv")


def progress_path(backend_name, family_name):
    return os.path.join(PROGRESS_DIR, f"{backend_name}_{family_name}.progress.json")


def load_progress(backend_name, family_name):
    path = progress_path(backend_name, family_name)
    if not os.path.exists(path):
        return {"completed_iterations": 0}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(backend_name, family_name, completed_iterations):
    path = progress_path(backend_name, family_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"completed_iterations": int(completed_iterations)}, f, indent=2)


def ensure_csv_header(path):
    if not os.path.exists(path):
        pd.DataFrame(columns=["POF", "ODR", "POS", "Target Value", "circuit"]).to_csv(path, index=False)


def append_rows(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["POF", "ODR", "POS", "Target Value", "circuit"])
    df.to_csv(path, mode="a", header=False, index=False)


def rows_for_one_iteration(circuit, backend_ideal, backend_noise, inp, iteration, qasm_dir):
    rows = []
    try:
        circuit.get_result(backend_ideal, inp, use_gpu=True)
        with open(
            os.path.join(qasm_dir, f"{circuit.key_aurguments['ID']}_{iteration}.qasm"),
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:
            f.write(circuit.key_aurguments["circuit"].qasm())
    except MissingOptionalLibraryError as ex:
        print(ex)

    ideal = circuit.get_result(backend_ideal, inp, use_gpu=True)
    noise = circuit.get_result(backend_noise, inp, use_gpu=True)
    ideal_bins = {x["bin"] for x in ideal["probability"]}

    for outputs in ideal["probability"]:
        target_variable_prob = None
        target_variable_odds = None
        actual_target_prob = outputs["prob"]
        all_other_probs = 0
        for noise_outputs in noise["probability"]:
            if outputs["bin"] == noise_outputs["bin"]:
                target_variable_prob = noise_outputs["prob"]
                target_variable_odds = noise_outputs["odds"]
            else:
                all_other_probs += noise_outputs["prob"]
        rows.append([
            all_other_probs,
            target_variable_odds,
            target_variable_prob,
            actual_target_prob,
            f"{circuit.key_aurguments['ID']}_{iteration}",
        ])

    for outputs in noise["probability"]:
        if outputs["bin"] not in ideal_bins:
            target_variable_prob = None
            target_variable_odds = None
            actual_target_prob = 0
            all_other_probs = 0
            for noise_outputs in noise["probability"]:
                if outputs["bin"] == noise_outputs["bin"]:
                    target_variable_prob = noise_outputs["prob"]
                    target_variable_odds = noise_outputs["odds"]
                else:
                    all_other_probs += noise_outputs["prob"]
            rows.append([
                all_other_probs,
                target_variable_odds,
                target_variable_prob,
                actual_target_prob,
                f"{circuit.key_aurguments['ID']}_{iteration}",
            ])
    return rows


def process_backend_family(circuit, family_name, backend_ideal, backend_noise, inputs, backend_name):
    out_csv = csv_output_path(backend_name, family_name)
    prog = load_progress(backend_name, family_name)
    start_iter = int(prog.get("completed_iterations", 0))

    ensure_csv_header(out_csv)

    if start_iter >= len(inputs):
        print(f"[skip] {family_name} on {backend_name}: already completed ({start_iter}/{len(inputs)})")
        return

    qasm_dir = os.path.join(BASELINE_TUNING_DIR, backend_name)
    os.makedirs(qasm_dir, exist_ok=True)

    print(f"[resume] {family_name} on {backend_name}: start from iteration {start_iter}/{len(inputs)}")
    for iteration in tqdm(range(start_iter, len(inputs))):
        inp = inputs[iteration]
        rows = rows_for_one_iteration(circuit, backend_ideal, backend_noise, inp, iteration, qasm_dir)
        append_rows(out_csv, rows)
        save_progress(backend_name, family_name, iteration + 1)


def filtered_backends():
    items = BACKENDS
    if BACKEND_FILTER:
        items = [x for x in items if x[0] == BACKEND_FILTER]
    if MAX_BACKENDS > 0:
        items = items[:MAX_BACKENDS]
    return items


def main():
    ensure_dirs()

    backend_factory = BackendFactory()
    backend_ideal = backend_factory.initialize_backend()

    split_payload = load_or_build_manifest(backend_ideal)

    print("CUTs:", split_payload["CUTs"])
    print("Evaluation counts by family:", split_payload["eval_counts"])
    print("Evaluation total:", split_payload["actual_eval_total"])

    selected_backends = filtered_backends()
    print("Backends to run:", [b for b, _ in selected_backends])

    backend_executors = {bk: backend_factory.initialize_backend(bk) for bk, _ in selected_backends}

    for family_name in split_payload["CUTs"]:
        if FAMILY_FILTER and family_name != FAMILY_FILTER:
            continue

        train_inputs = split_payload["families"][family_name]["train_inputs"]
        print(f"Current circuit family: {family_name} | train circuits={len(train_inputs)}")
        circuit = get_circuit_class_object(family_name)

        for bk, _ in selected_backends:
            print(f"Generating Data For {bk} Backend")
            print("------------------------------------------")
            process_backend_family(
                circuit,
                family_name,
                backend_ideal,
                backend_executors[bk],
                train_inputs,
                bk,
            )
            # train_predictor(csv_output_path(bk, family_name), bk, family_name)


if __name__ == "__main__":
    main()
