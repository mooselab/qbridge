import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
QOIN_DIR = os.path.join(PROJECT_ROOT, "QOIN")

if QOIN_DIR not in sys.path:
    sys.path.insert(0, QOIN_DIR)

from benchmark_circuits import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import exrex
import math

from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.converters import circuit_to_dag, dag_to_circuit

warnings.filterwarnings("ignore")


# =========================
# Config
# =========================
SHOTS = 1024
SEED = 1997
TRAIN_SPLIT = 0.4
RANDOM_STATE = 13

BACKENDS = [
    ('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65),
    ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28),
    ('FakeCasablanca', 7), ('FakeGuadalupe', 16), ('FakeHanoi', 27),
    ('FakeJakarta', 7), ('FakeJohannesburg', 20), ('FakeKolkata', 27),
    ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27),
    ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27),
    ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27),
    ('FakeToronto', 27), ('FakeWashington', 127)
]

OUTPUT_ROOT = "./data/qlear_pretrain_by_backend"


# =========================
# Helpers copied/adapted from original script
# =========================
def getIntegers(startRange, upperlimit):
    return [x for x in range(startRange, startRange + upperlimit)]


def getallBinaryValues(number_bits):
    max_number = np.power(2, number_bits) - 1
    data = []
    for x in range(0, max_number):
        value = bin(x).replace("0b", "")
        value = "0" * (number_bits - len(value)) + value
        data.append(value)
    return data


def getexpression(regex, upperlimit):
    return [exrex.getone(regex) for _ in range(upperlimit)]


def generate_data(fmt, startRange, endRange, percentage, regex, circuit):
    data = []
    if fmt == "int":
        upperlimit = int(np.ceil((endRange - startRange) * percentage))
        data = getIntegers(startRange, upperlimit)
    elif fmt == "binary":
        upperlimit = int(np.ceil((endRange - startRange) * percentage))
        data = getallBinaryValues(startRange + upperlimit)
    elif fmt == "expression":
        upperlimit = int(np.ceil((math.factorial(startRange)) * percentage))
        data = getexpression(regex, upperlimit)
    return data, circuit


def read_configuration(filepath):
    with open(filepath, "r") as file:
        content = file.read()
    rules = content.split("-" * 20)
    rules_dict = {}
    for rule in rules:
        try:
            parameters = rule.split("\n")
            parameter_dict = {}
            for parameter in parameters:
                if parameter != "":
                    key, value = parameter.split(":")
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except Exception:
                            pass
                    parameter_dict[key] = value
            rules_dict[parameter_dict.pop("ID")] = parameter_dict
        except Exception:
            continue
    return rules_dict


def normalize_prob_dict(prob_dict):
    total = sum(prob_dict.values())
    if total <= 0:
        return prob_dict
    return {k: v / total for k, v in prob_dict.items()}


def probs_to_dict(result_prob_list):
    out = {}
    for item in result_prob_list:
        out[str(item["bin"])] = float(item["prob"])
    return normalize_prob_dict(out)


def odds_from_prob(p):
    if p >= 1.0:
        return 1e12
    if p <= 0.0:
        return 0.0
    return p / (1.0 - p)


def state_weight(bitstr):
    return sum(1 for ch in str(bitstr) if ch == "1")


def hellinger_distance_dict(ideal_probs, noisy_probs):
    all_keys = sorted(set(ideal_probs.keys()) | set(noisy_probs.keys()))
    s = 0.0
    for k in all_keys:
        p = max(0.0, float(ideal_probs.get(k, 0.0)))
        q = max(0.0, float(noisy_probs.get(k, 0.0)))
        s += (np.sqrt(p) - np.sqrt(q)) ** 2
    return (1.0 / np.sqrt(2.0)) * np.sqrt(s)


def get_measured_qubits_from_original(circuit: QuantumCircuit):
    measured = []
    for inst, qargs, _ in circuit.data:
        if inst.name == "measure":
            for q in qargs:
                measured.append(circuit.find_bit(q).index)
    return sorted(set(measured))


def remove_measurements_and_barriers(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Remove final measurements and barriers so DPE is computed on the computational circuit.
    """
    base = circuit.remove_final_measurements(inplace=False)
    new_circ = QuantumCircuit(base.num_qubits, name=base.name)

    for inst, qargs, _ in base.data:
        if inst.name == "barrier":
            continue
        qidx = [base.find_bit(q).index for q in qargs]
        qubits = [new_circ.qubits[i] for i in qidx]
        new_circ.append(inst, qubits, [])

    return new_circ


def split_transpiled_circuit_into_depth_layers(transpiled_no_meas: QuantumCircuit):
    """
    Split a transpiled circuit into true DAG depth layers.
    """
    dag = circuit_to_dag(transpiled_no_meas)
    layer_circuits = []

    for layer in dag.layers():
        layer_circ = dag_to_circuit(layer["graph"])

        cleaned = QuantumCircuit(transpiled_no_meas.num_qubits)
        non_barrier_count = 0

        for inst, qargs, _ in layer_circ.data:
            if inst.name == "barrier":
                continue
            qidx = [layer_circ.find_bit(q).index for q in qargs]
            qubits = [cleaned.qubits[i] for i in qidx]
            cleaned.append(inst, qubits, [])
            non_barrier_count += 1

        if non_barrier_count > 0:
            layer_circuits.append(cleaned)

    return layer_circuits


def compose_layer_range(num_qubits: int, layer_circuits, start_layer: int, end_layer: int):
    sub = QuantumCircuit(num_qubits)
    for i in range(start_layer, end_layer):
        sub.compose(layer_circuits[i], inplace=True)
    return sub


def build_strict_depth_cut_subcircuits(transpiled_circuit: QuantumCircuit):
    """
    Strict paper-style depth cuts on the transpiled circuit:
      Q1: [0, 1/4 depth)
      Q2: [1/4, 1/2 depth)
      Q3: [1/2, 3/4 depth)
    """
    no_meas = remove_measurements_and_barriers(transpiled_circuit)
    layer_circuits = split_transpiled_circuit_into_depth_layers(no_meas)

    total_depth = len(layer_circuits)
    if total_depth == 0:
        return {
            "Q1": None,
            "Q2": None,
            "Q3": None,
            "depth": 0,
            "bounds": (0, 0, 0, 0),
        }

    b1 = max(1, int(np.floor(total_depth * 0.25)))
    b2 = max(b1 + 1, int(np.floor(total_depth * 0.50)))
    b3 = max(b2 + 1, int(np.floor(total_depth * 0.75)))

    b1 = min(b1, total_depth)
    b2 = min(b2, total_depth)
    b3 = min(b3, total_depth)

    q1 = compose_layer_range(no_meas.num_qubits, layer_circuits, 0, b1) if b1 > 0 else None
    q2 = compose_layer_range(no_meas.num_qubits, layer_circuits, b1, b2) if b2 > b1 else None
    q3 = compose_layer_range(no_meas.num_qubits, layer_circuits, b2, b3) if b3 > b2 else None

    return {
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "depth": total_depth,
        "bounds": (0, b1, b2, b3),
    }


def build_qn_qn_dagger_circuit(subcirc: QuantumCircuit, measured_qubits):
    """
    Build Q_n Q_n^\dagger and measure only original measured qubits.
    """
    if subcirc is None:
        return None
    if len(measured_qubits) == 0:
        return None

    n_qubits = subcirc.num_qubits
    n_meas = len(measured_qubits)

    qc = QuantumCircuit(n_qubits, n_meas)
    qc.compose(subcirc, inplace=True)
    qc.compose(subcirc.inverse(), inplace=True)

    for cidx, qidx in enumerate(measured_qubits):
        qc.measure(qidx, cidx)

    return qc


def run_distribution_for_circuit(circuit: QuantumCircuit, backend, shots=1024, seed=1997):
    tqc = transpile(circuit, backend=backend, optimization_level=0, seed_transpiler=seed)
    result = backend.run(tqc, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {str(k): v / total for k, v in counts.items()}


def count_1q_2q_gates(transpiled_circuit: QuantumCircuit):
    n1 = 0
    n2 = 0
    for inst, qargs, _ in transpiled_circuit.data:
        if inst.name in {"measure", "barrier"}:
            continue
        nq = len(qargs)
        if nq == 1:
            n1 += 1
        elif nq == 2:
            n2 += 1
    return n1, n2


def extract_circuit_features_from_transpiled(transpiled_circuit: QuantumCircuit):
    n1, n2 = count_1q_2q_gates(transpiled_circuit)
    return {
        "circuit_width": transpiled_circuit.num_qubits,
        "circuit_depth": transpiled_circuit.depth(),
        "Num_1Q_Gates": n1,
        "Num_2Q_Gates": n2,
        "transpiled_circuit": transpiled_circuit,
    }


def compute_dpe_features_strict(raw_circuit: QuantumCircuit, noisy_backend, shots=1024, seed=1997):
    """
    Strict DPE:
      1. Transpile raw circuit to target backend.
      2. Remove measurements.
      3. Split by true transpiled depth layers.
      4. Build Q1Q1^dagger, Q2Q2^dagger, Q3Q3^dagger.
      5. Measure original measured qubits only.
      6. Compute Hellinger distance to ideal all-zero distribution.
    """
    measured_qubits = get_measured_qubits_from_original(raw_circuit)

    transpiled_full = transpile(
        raw_circuit,
        backend=noisy_backend,
        optimization_level=0,
        seed_transpiler=seed
    )

    cuts = build_strict_depth_cut_subcircuits(transpiled_full)

    feature_map = {
        "Avg_inverted_error_25": np.nan,
        "Avg_inverted_error_50": np.nan,
        "Avg_inverted_error_75": np.nan,
    }

    qcircs = [
        ("Avg_inverted_error_25", build_qn_qn_dagger_circuit(cuts["Q1"], measured_qubits)),
        ("Avg_inverted_error_50", build_qn_qn_dagger_circuit(cuts["Q2"], measured_qubits)),
        ("Avg_inverted_error_75", build_qn_qn_dagger_circuit(cuts["Q3"], measured_qubits)),
    ]

    for key, qc in qcircs:
        if qc is None:
            continue

        noisy_probs = run_distribution_for_circuit(qc, backend=noisy_backend, shots=shots, seed=seed)
        ideal_zero = "0" * qc.num_clbits
        ideal_probs = {ideal_zero: 1.0}
        feature_map[key] = hellinger_distance_dict(ideal_probs, noisy_probs)

    return feature_map, transpiled_full


def build_rows_for_distribution(
    circuit_name,
    sample_id,
    ideal_probs,
    noisy_probs,
    circuit_feature_dict,
    dpe_feature_dict
):
    all_states = sorted(ideal_probs.keys())
    rows = []

    for bitstr in all_states:
        ideal_p = float(ideal_probs.get(bitstr, 0.0))
        noisy_p = float(noisy_probs.get(bitstr, 0.0))
        row = {
            "circuit_family": circuit_name,
            "circuit": sample_id,
            "output_state": bitstr,
            "target": ideal_p,
            "observed_prob_50": noisy_p,
            "Avg_odds_ratio": odds_from_prob(noisy_p),
            "state_weight": state_weight(bitstr),
            "circuit_width": circuit_feature_dict["circuit_width"],
            "circuit_depth": circuit_feature_dict["circuit_depth"],
            "Num_1Q_Gates": circuit_feature_dict["Num_1Q_Gates"],
            "Num_2Q_Gates": circuit_feature_dict["Num_2Q_Gates"],
        }
        row.update(dpe_feature_dict)
        rows.append(row)

    return rows


# =========================
# Main generation pipeline
# =========================
def prepare_pretrain_circuits():
    baseline_circuits, _ = train_test_split(
        get_all_circuits(),
        train_size=TRAIN_SPLIT,
        random_state=RANDOM_STATE
    )

    rules = read_configuration("../QOIN/Configuration.txt")
    data_circuit_pairs = []

    for baseline_circuit in baseline_circuits:
        circuit_obj = get_circuit_class_object(baseline_circuit)
        print(f"Preparing baseline circuit family: {baseline_circuit}")
        rule = rules[circuit_obj.key_aurguments["ID"]]
        data, circuit_obj = generate_data(
            fmt=rule["FORMAT"],
            startRange=rule["START"],
            endRange=rule["END"],
            percentage=rule["PERCENTAGE"],
            regex=rule["REGEX"],
            circuit=circuit_obj
        )
        data_circuit_pairs.append((baseline_circuit, data, circuit_obj))

    return data_circuit_pairs


def iter_inputs_for_circuit(circuit_obj, data):
    if circuit_obj.key_aurguments["input_type"] == 1:
        for iteration in range(int(np.power(len(data), 2))):
            yield iteration, data[iteration % len(data)]
    elif circuit_obj.key_aurguments["input_type"] == 2:
        pairs = [[x, y] for y in data for x in data]
        for iteration, inp in enumerate(pairs):
            yield iteration, inp
    else:
        raise ValueError(f"Unsupported input_type: {circuit_obj.key_aurguments['input_type']}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    backend_factory = BackendFactory()
    ideal_backend = backend_factory.initialize_backend()

    noisy_backends = {}
    print("Initializing fake backends...")
    for bk, _ in tqdm(BACKENDS):
        noisy_backends[bk] = backend_factory.initialize_backend(bk)

    data_circuit_pairs = prepare_pretrain_circuits()

    for bk, _ in BACKENDS:
        print(f"\nGenerating Q-LEAR pretraining features for backend: {bk}")
        print("-" * 60)

        backend_dir = os.path.join(OUTPUT_ROOT, bk)
        os.makedirs(backend_dir, exist_ok=True)

        all_rows = []

        for circuit_name, data, circuit_obj in data_circuit_pairs:
            print(f"Processing circuit family: {circuit_name}")

            for iteration, inp in tqdm(list(iter_inputs_for_circuit(circuit_obj, data)), desc=f"{bk}:{circuit_name}"):
                sample_id = f"{circuit_obj.key_aurguments['ID']}_{iteration}"

                try:
                    ideal_result = circuit_obj.get_result(
                        ideal_backend, inp, number_of_runs=SHOTS, seed=SEED
                    )
                    noisy_result = circuit_obj.get_result(
                        noisy_backends[bk], inp, number_of_runs=SHOTS, seed=SEED
                    )
                except MissingOptionalLibraryError as ex:
                    print(f"[Skip] Missing optional library for {sample_id}: {ex}")
                    continue
                except Exception as ex:
                    print(f"[Skip] Failed execution for {sample_id}: {ex}")
                    continue

                raw_circuit = circuit_obj.key_aurguments.get("circuit", None)
                if raw_circuit is None:
                    print(f"[Skip] No generated circuit found for {sample_id}")
                    continue

                try:
                    dpe_features, transpiled_full = compute_dpe_features_strict(
                        raw_circuit,
                        noisy_backends[bk],
                        shots=SHOTS,
                        seed=SEED
                    )
                    circuit_features = extract_circuit_features_from_transpiled(transpiled_full)
                except Exception as ex:
                    print(f"[Skip] Feature extraction failed for {sample_id}: {ex}")
                    continue

                ideal_probs = probs_to_dict(ideal_result["probability"])
                noisy_probs = probs_to_dict(noisy_result["probability"])

                rows = build_rows_for_distribution(
                    circuit_name=circuit_name,
                    sample_id=sample_id,
                    ideal_probs=ideal_probs,
                    noisy_probs=noisy_probs,
                    circuit_feature_dict=circuit_features,
                    dpe_feature_dict=dpe_features
                )
                all_rows.extend(rows)

                try:
                    qasm_path = os.path.join(backend_dir, f"{sample_id}.qasm")
                    with open(qasm_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(raw_circuit.qasm())
                except Exception:
                    pass

        df = pd.DataFrame(all_rows)

        preferred_cols = [
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
            "target",
            "circuit",
            "circuit_family",
            "output_state",
        ]
        existing_cols = [c for c in preferred_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in existing_cols]
        df = df[existing_cols + extra_cols]

        out_csv = os.path.join(backend_dir, f"{bk}_qlear_pretrain.csv")
        df.to_csv(out_csv, index=False)

        flat_out = os.path.join(OUTPUT_ROOT, f"{bk}.csv")
        df.to_csv(flat_out, index=False)

        print(f"Saved: {out_csv}")
        print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
