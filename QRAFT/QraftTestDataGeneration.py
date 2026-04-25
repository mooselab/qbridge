import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
QOIN_DIR = os.path.join(PROJECT_ROOT, "QOIN")

if QOIN_DIR not in sys.path:
    sys.path.insert(0, QOIN_DIR)
import copy
import random
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from qiskit.exceptions import MissingOptionalLibraryError

from benchmark_circuits import *
from QraftFeatureGeneration import extract_qraft_rows_for_circuit

warnings.filterwarnings("ignore")

# -------------------------
# Config
# -------------------------
T_RUNS = 10
SHOTS = 1024

OUTPUT_ROOT = "./data/qraft_test_data"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

backends = [
    ('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65),
    ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28),
    ('FakeCasablanca', 7), ('FakeGuadalupe', 16), ('FakeHanoi', 27),
    ('FakeJakarta', 7), ('FakeJohannesburg', 20), ('FakeKolkata', 27),
    ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27),
    ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27),
    ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27),
    ('FakeToronto', 27), ('FakeWashington', 127)
]

BaselineCircuits, CUTs = train_test_split(get_all_circuits(), train_size=0.4, random_state=13)

backend_factory = BackendFactory()
ideal_backend = backend_factory.initialize_backend()
backend_executors = {bk: backend_factory.initialize_backend(bk) for bk, _ in backends}


def safe_family_name(x):
    s = str(x)
    s = s.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return s


for backend_idx, (bk, qubit_size) in enumerate(backends):
    print(f"\n==============================")
    print(f"Generating Qraft TEST data for backend: {bk}")
    print(f"==============================")

    backend_dir = os.path.join(OUTPUT_ROOT, bk)
    os.makedirs(backend_dir, exist_ok=True)

    for cut in CUTs:
        family_name = safe_family_name(cut)
        print(f"\nCurrent family: {family_name}")

        circuit_template = get_circuit_class_object(cut)
        test_inputs = circuit_template.get_inputs()

        full_rows = []
        model_rows = []

        for iteration, inp in enumerate(tqdm(test_inputs, desc=f"{bk}-{family_name}", leave=False)):
            circuit = copy.deepcopy(circuit_template)

            try:
                _ = circuit.get_result(ideal_backend, inp)
            except MissingOptionalLibraryError as ex:
                print(ex)
                continue
            except Exception as ex:
                print(f"[WARN] instantiate failed | backend={bk} family={family_name} iter={iteration} inp={inp} :: {ex}")
                continue

            qc = circuit.key_aurguments.get("circuit", None)
            if qc is None:
                print(f"[WARN] no circuit object | backend={bk} family={family_name} iter={iteration}")
                continue

            if qc.num_qubits > qubit_size:
                continue

            circuit_id = f"{family_name}_{iteration}"

            try:
                qfull, qmodel = extract_qraft_rows_for_circuit(
                    qc_raw=qc,
                    backend_name=bk,
                    backend_obj=backend_executors[bk],
                    computer_id=backend_idx,
                    circuit_id=circuit_id,
                    t_runs=T_RUNS,
                    shots=SHOTS,
                )
            except MissingOptionalLibraryError as ex:
                print(ex)
                continue
            except Exception as ex:
                print(f"[WARN] feature extraction failed | backend={bk} family={family_name} iter={iteration} :: {ex}")
                continue

            for row in qfull:
                row["Family"] = family_name
                row["FamilyID"] = circuit.key_aurguments.get("ID", -1)
                row["InputInstance"] = str(inp)
            full_rows.extend(qfull)

            for row in qmodel:
                row["Family"] = family_name
                row["FamilyID"] = circuit.key_aurguments.get("ID", -1)
                row["CircuitID"] = circuit_id
            model_rows.extend(qmodel)

        if len(full_rows) == 0:
            print(f"[WARN] no test rows generated for backend={bk}, family={family_name}")
            continue

        df_full = pd.DataFrame(full_rows)
        df_model = pd.DataFrame(model_rows)

        full_path = os.path.join(backend_dir, f"{bk}_{family_name}_qraft_full.csv")
        model_path = os.path.join(backend_dir, f"{bk}_{family_name}_qraft_model.csv")

        df_full.to_csv(full_path, index=False)
        df_model.to_csv(model_path, index=False)

        print(f"Saved: {full_path}")
        print(f"Saved: {model_path}")

print("\nAll Qraft test data generated.")