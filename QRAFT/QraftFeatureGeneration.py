import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
QOIN_DIR = os.path.join(PROJECT_ROOT, "QOIN")

if QOIN_DIR not in sys.path:
    sys.path.insert(0, QOIN_DIR)

from benchmark_circuits import *
import copy
import math
import os
import random
import warnings

import exrex
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.exceptions import MissingOptionalLibraryError

warnings.filterwarnings("ignore")


# =========================
# User-facing configuration
# =========================

T_RUNS = 10
SHOTS = 1024
OUTPUT_DIR = "./data/qraft_training_data"
CONFIG_PATH = "../QOIN/Configuration.txt"
TRAIN_SIZE = 0.4
RANDOM_STATE = 13

# Keep the same backend list/order as your DataGeneration.py.
# ComputerID will be the index in this list.
BACKENDS = [
    ("FakeAlmaden", 20),
    ("FakeBoeblingen", 20),
    ("FakeBrooklyn", 65),
    ("FakeCairo", 27),
    ("FakeCambridge", 28),
    ("FakeCambridgeAlternativeBasis", 28),
    ("FakeCasablanca", 7),
    ("FakeGuadalupe", 16),
    ("FakeHanoi", 27),
    ("FakeJakarta", 7),
    ("FakeJohannesburg", 20),
    ("FakeKolkata", 27),
    ("FakeLagos", 7),
    ("FakeManhattan", 65),
    ("FakeMontreal", 27),
    ("FakeMumbai", 27),
    ("FakeNairobi", 7),
    ("FakeParis", 27),
    ("FakeRochester", 53),
    ("FakeSingapore", 20),
    ("FakeSydney", 27),
    ("FakeToronto", 27),
    ("FakeWashington", 127),
]


# =========================
# Helpers copied/adapted from DataGeneration.py
# =========================

def getIntegers(startRange, upperlimit):
    return [x for x in range(startRange, startRange + upperlimit)]


def getallBinaryValues(number_bits):
    # Fixed version: include the all-ones state
    max_number = int(np.power(2, number_bits) - 1)
    data = []
    for x in range(0, max_number + 1):
        value = bin(x).replace("0b", "")
        value = "0" * (number_bits - len(value)) + value
        data.append(value)
    return data


def getexpression(regex, upperlimit):
    return [exrex.getone(regex) for _ in range(upperlimit)]


def generate_data(Format, startRange, endRange, percentage, regex, circuit):
    data = []
    if Format == "int":
        upperlimit = int(np.ceil((endRange - startRange) * percentage))
        data = getIntegers(startRange, upperlimit)
    if Format == "binary":
        upperlimit = int(np.ceil((endRange - startRange) * percentage))
        data = getallBinaryValues(startRange + upperlimit)
    if Format == "expression":
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


# =========================
# Qraft feature helpers
# =========================

def all_bitstrings(n):
    return [format(i, f"0{n}b") for i in range(2 ** n)]


def remove_measurements_if_any(qc: QuantumCircuit) -> QuantumCircuit:
    try:
        return qc.remove_final_measurements(inplace=False)
    except Exception:
        return qc.copy()


def add_all_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    qc2 = qc.copy()
    qc2.measure_all()
    return qc2


def resolve_backend(obj):
    """
    Try to recover a runnable qiskit backend from whatever BackendFactory returned.
    """
    if hasattr(obj, "run"):
        return obj

    candidate_attrs = [
        "backend",
        "_backend",
        "backend_obj",
        "simulator",
        "_simulator",
        "executor",
        "_executor",
    ]
    for attr in candidate_attrs:
        if hasattr(obj, attr):
            candidate = getattr(obj, attr)
            if hasattr(candidate, "run"):
                return candidate

    raise TypeError(
        f"Could not resolve a runnable backend from object of type {type(obj)}"
    )


def probs_from_counts(counts, n, shots):
    out = {b: 0.0 for b in all_bitstrings(n)}
    for bitstr, c in counts.items():
        clean = str(bitstr).replace(" ", "")
        if len(clean) > n:
            clean = clean[-n:]
        out[clean] = c / shots
    return out


def ideal_probabilities(fc_nom: QuantumCircuit):
    """
    Exact ideal output distribution of the forward circuit.
    """
    sv = Statevector.from_instruction(fc_nom)
    probs = sv.probabilities_dict()

    n = fc_nom.num_qubits
    out = {b: 0.0 for b in all_bitstrings(n)}
    for bitstr, p in probs.items():
        clean = str(bitstr).replace(" ", "")
        if len(clean) > n:
            clean = clean[-n:]
        out[clean] = float(np.real_if_close(p))
    return out


def qraft_compatible_gate_counts(transpiled_qc: QuantumCircuit):
    """
    Count gates in a Qraft-compatible 4-bucket format.

    Buckets:
      - u1: single-qubit gates with 1 effective parameter
      - u2: single-qubit gates with 2 effective parameters
      - u3: single-qubit gates with 3 effective parameters
      - cx: two-qubit entangling gates

    Important:
    This is not trying to reproduce the exact 2021 gate names.
    It defines a consistent feature space for both training and inference.
    """
    counts = {"u1": 0, "u2": 0, "u3": 0, "cx": 0}

    for inst, qargs, cargs in transpiled_qc.data:
        name = inst.name
        n_qubits = len(qargs)

        if name == "barrier":
            continue
        if name == "measure":
            continue

        # Two-qubit entangling bucket
        if n_qubits == 2 and name in {"cx", "cz", "ecr"}:
            counts["cx"] += 1
            continue

        # One-qubit bucket by parameter count
        if n_qubits == 1:
            n_params = len(inst.params)

            if n_params == 1:
                counts["u1"] += 1
            elif n_params == 2:
                counts["u2"] += 1
            elif n_params >= 3:
                counts["u3"] += 1
            else:
                # Gates like x, y, z, h, sx, s, t have 0 explicit params.
                # Put them in u3-like bucket as generic 1q operations.
                counts["u3"] += 1

    return counts


def run_probabilities_many_times(measured_qc, backend, t_runs, shots, initial_layout=None):
    """
    Transpile once, run many times, return a list of per-run probability dicts.
    """
    n = measured_qc.num_qubits
    tqc = transpile(
        measured_qc,
        backend=backend,
        initial_layout=initial_layout,
        optimization_level=3,
    )

    runs = []
    for _ in range(t_runs):
        job = backend.run(tqc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        runs.append(probs_from_counts(counts, n, shots))
    return tqc, runs


def percentile(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q))


def extract_qraft_rows_for_circuit(
    qc_raw: QuantumCircuit,
    backend_name: str,
    backend_obj,
    computer_id: int,
    circuit_id: str,
    t_runs: int = T_RUNS,
    shots: int = SHOTS,
):
    """
    Given one instantiated circuit, generate all per-state Qraft rows.
    Returns:
      full_rows: rows with metadata
      model_rows: rows matching the old inputData.csv schema
    """
    fc_nom = sanitize_qraft_base_circuit(qc_raw)
    n = fc_nom.num_qubits

    if n == 0:
        return [], []

    if len(fc_nom.data) == 0:
        return [], []

    initial_layout = list(range(n))
    frc_nom = fc_nom.compose(fc_nom.inverse())

    fc_meas = add_all_measurements(fc_nom)
    frc_meas = add_all_measurements(frc_nom)

    runnable_backend = resolve_backend(backend_obj)

    transpiled_fc, fc_runs = run_probabilities_many_times(
        fc_meas,
        runnable_backend,
        t_runs=t_runs,
        shots=shots,
        initial_layout=initial_layout,
    )
    _, frc_runs = run_probabilities_many_times(
        frc_meas,
        runnable_backend,
        t_runs=t_runs,
        shots=shots,
        initial_layout=initial_layout,
    )

    gate_counts = qraft_compatible_gate_counts(transpiled_fc)
    width = int(transpiled_fc.num_qubits)
    depth = int(transpiled_fc.depth())

    ideal_probs = ideal_probabilities(fc_nom)
    states = all_bitstrings(n)
    zero_state = "0" * n

    total_updn_err_runs = [1.0 - run[zero_state] for run in frc_runs]
    total_updn_err25 = percentile(total_updn_err_runs, 25)
    total_updn_err50 = percentile(total_updn_err_runs, 50)
    total_updn_err75 = percentile(total_updn_err_runs, 75)

    full_rows = []
    model_rows = []

    for state in states:
        hw = state.count("1")

        up_vals = [run[state] for run in fc_runs]
        up_prob25 = percentile(up_vals, 25)
        up_prob50 = percentile(up_vals, 50)
        up_prob75 = percentile(up_vals, 75)

        updn_vals = [run[state] for run in frc_runs]
        if state == zero_state:
            updn_err_vals = [v - 1.0 for v in updn_vals]
        else:
            updn_err_vals = updn_vals

        updn_err25 = percentile(updn_err_vals, 25)
        updn_err50 = percentile(updn_err_vals, 50)
        updn_err75 = percentile(updn_err_vals, 75)

        state_real_prob = float(ideal_probs.get(state, 0.0))

        full_row = {
            "CircuitID": circuit_id,
            "BackendName": backend_name,
            "ComputerID": computer_id,
            "CircuitWidth": width,
            "CircuitDepth": depth,
            "CircuitNumU1Gates": gate_counts["u1"],
            "CircuitNumU2Gates": gate_counts["u2"],
            "CircuitNumU3Gates": gate_counts["u3"],
            "CircuitNumCXGates": gate_counts["cx"],
            "TotalUpDnErr25": total_updn_err25,
            "TotalUpDnErr50": total_updn_err50,
            "TotalUpDnErr75": total_updn_err75,
            "State": state,
            "StateHammingWeight": hw,
            "StateUpProb25": up_prob25,
            "StateUpProb50": up_prob50,
            "StateUpProb75": up_prob75,
            "StateUpDnErr25": updn_err25,
            "StateUpDnErr50": updn_err50,
            "StateUpDnErr75": updn_err75,
            "StateRealProb": state_real_prob,
        }
        full_rows.append(full_row)

        model_row = {
            "ComputerID": computer_id,
            "CircuitWidth": width,
            "CircuitDepth": depth,
            "CircuitNumU1Gates": gate_counts["u1"],
            "CircuitNumU2Gates": gate_counts["u2"],
            "CircuitNumU3Gates": gate_counts["u3"],
            "CircuitNumCXGates": gate_counts["cx"],
            "TotalUpDnErr25": int(round(total_updn_err25 * 100)),
            "TotalUpDnErr50": int(round(total_updn_err50 * 100)),
            "TotalUpDnErr75": int(round(total_updn_err75 * 100)),
            "StateHammingWeight": hw,
            "StateUpProb25": int(round(up_prob25 * 100)),
            "StateUpProb50": int(round(up_prob50 * 100)),
            "StateUpProb75": int(round(up_prob75 * 100)),
            "StateUpDnErr25": int(round(updn_err25 * 100)),
            "StateUpDnErr50": int(round(updn_err50 * 100)),
            "StateUpDnErr75": int(round(updn_err75 * 100)),
            "StateRealProb": int(round(state_real_prob * 100)),
        }
        model_rows.append(model_row)

    return full_rows, model_rows

def sanitize_qraft_base_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    clean = QuantumCircuit(*qc.qregs, *qc.cregs)

    for inst, qargs, cargs in qc.data:
        name = inst.name
        if name in {"measure", "barrier", "reset"}:
            continue
        clean.append(inst, qargs, cargs)

    return clean


# =========================
# Main generation logic
# =========================

def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Select baseline circuits exactly as before
    BaselineCircuits, _ = train_test_split(
        get_all_circuits(),
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
    )

    # Generate per-circuit input domains exactly as before
    rules = read_configuration(CONFIG_PATH)
    data_circuit_pairs = []
    for baseline_circuit in BaselineCircuits:
        circuit = get_circuit_class_object(baseline_circuit)
        print(f"Generating data for baseline circuit: {baseline_circuit}")
        rule = rules[circuit.key_aurguments["ID"]]
        data, circuit = generate_data(
            Format=rule["FORMAT"],
            startRange=rule["START"],
            endRange=rule["END"],
            percentage=rule["PERCENTAGE"],
            regex=rule["REGEX"],
            circuit=circuit,
        )
        data_circuit_pairs.append((data, circuit))

    # Initialize ideal/noisy backends exactly as before
    backend_factory = BackendFactory()
    ideal_executor = backend_factory.initialize_backend()
    backend_executors = {}

    print("Initializing backends")
    for bk, _ in tqdm(BACKENDS):
        backend_executors[bk] = backend_factory.initialize_backend(bk)

    # Per-backend CSVs
    for computer_id, (bk, qubit_size) in enumerate(BACKENDS):
        print(f"\nGenerating Qraft features for backend: {bk}")
        print("-" * 60)

        backend_full_rows = []
        backend_model_rows = []

        for data, circuit_template in data_circuit_pairs:
            print(f"Baseline circuit ID: {circuit_template.key_aurguments['ID']}")

            input_type = circuit_template.key_aurguments["input_type"]

            if input_type == 1:
                inputs = list(data)
            elif input_type == 2:
                inputs = [[x, y] for y in data for x in data]
            else:
                raise ValueError(f"Unsupported input_type={input_type}")

            for iteration, inp in enumerate(tqdm(inputs, leave=False)):
                circuit = copy.deepcopy(circuit_template)

                try:
                    # This call is used only to instantiate/update the circuit object
                    _ = circuit.get_result(ideal_executor, inp)
                except MissingOptionalLibraryError as ex:
                    print(ex)
                    continue
                except Exception as ex:
                    print(f"[WARN] Failed to instantiate circuit {circuit.key_aurguments['ID']} with input {inp}: {ex}")
                    continue

                qc = circuit.key_aurguments.get("circuit", None)
                if qc is None:
                    print(f"[WARN] No circuit object found for {circuit.key_aurguments['ID']} input {inp}")
                    continue

                if qc.num_qubits > qubit_size:
                    continue

                circuit_id = f"{circuit.key_aurguments['ID']}_{iteration}"

                try:
                    full_rows, model_rows = extract_qraft_rows_for_circuit(
                        qc_raw=qc,
                        backend_name=bk,
                        backend_obj=backend_executors[bk],
                        computer_id=computer_id,
                        circuit_id=circuit_id,
                        t_runs=T_RUNS,
                        shots=SHOTS,
                    )
                except MissingOptionalLibraryError as ex:
                    print(ex)
                    continue
                except Exception as ex:
                    print(f"[WARN] Failed Qraft extraction for {circuit_id} on {bk}: {ex}")
                    continue

                backend_full_rows.extend(full_rows)
                backend_model_rows.extend(model_rows)

        # Save both full and model-only CSVs
        df_full = pd.DataFrame(backend_full_rows)
        df_model = pd.DataFrame(backend_model_rows)

        full_path = os.path.join(OUTPUT_DIR, f"{bk}_qraft_full.csv")
        model_path = os.path.join(OUTPUT_DIR, f"{bk}_qraft_model.csv")

        df_full.to_csv(full_path, index=False)
        df_model.to_csv(model_path, index=False)

        print(f"Saved: {full_path}")
        print(f"Saved: {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()