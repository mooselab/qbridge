import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import pickle
import random
import warnings
from collections import defaultdict
import ktrain
from ktrain import tabular
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import exrex
from qiskit.exceptions import MissingOptionalLibraryError

from benchmark_circuits import *

import tensorflow as tf

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

backends = [
    ('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65),
    ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28),
    ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJohannesburg', 20), ('FakeKolkata', 27),
    ('FakeManhattan', 65), ('FakeMontreal', 27), ('FakeMumbai', 27), ('FakeParis', 27),
    ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27),
    ('FakeToronto', 27), ('FakeWashington', 127)
]
BaselineCircuits,CUTs = train_test_split(get_all_circuits(),train_size=0.4,random_state=13)
SEEDS = [1, 2, 3, 4, 5]

def getIntegers(startRange,upperlimit):
    return [x for x in range(startRange,startRange+upperlimit)]

def getallBinaryValues(number_bits):
    max_number = np.power(2,number_bits)-1
    data = []
    for x in range(0,max_number):
        value = bin(x).replace("0b","")
        value = "0"*(number_bits-len(value))+value
        data.append(value)
    return data

def getexpression(regex,upperlimit):
    return [exrex.getone(regex) for x in range(upperlimit)]


def generate_data(Format,startRange,endRange,percentage,regex,circuit):
    data = []
    if Format=="int":
        upperlimit = int(np.ceil((endRange-startRange)*percentage))
        data = getIntegers(startRange,upperlimit)
    if Format=="binary":
        upperlimit = int(np.ceil((endRange-startRange)*percentage))
        data = getallBinaryValues(startRange+upperlimit)
    if Format=="expression":
        upperlimit = int(np.ceil((math.factorial(startRange))*percentage))
        data = getexpression(regex,upperlimit)
    return data,circuit

# # Get Evaluation data for CUTs
def read_configuration(filepath):
    with open(filepath,"r") as file:
        content = file.read()
    rules = content.split("-"*20)
    rules_dict = {}
    for rule in rules:
        try:
            parameters = rule.split("\n")
            parameter_dict = {}
            for parameter in parameters:
                if parameter!="":
                    key,value = parameter.split(":")
                    value = value.strip()
                    if value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except:
                            pass
                    parameter_dict[key] = value
            rules_dict[parameter_dict.pop("ID")] = parameter_dict
        except:
            continue
    return rules_dict


rules = read_configuration("Configuration.txt")
data_circuit_pairs = []
for baseline_circuit in CUTs:
    circuit = get_circuit_class_object(baseline_circuit)
    print("Generating data for CUT circuit: {}, ID:{}".format(baseline_circuit,circuit.key_aurguments["ID"]))
    rule = rules[circuit.key_aurguments["ID"]]
    data,circuit = generate_data(Format=rule["FORMAT"],startRange=rule["START"],
                                 endRange=rule["END"],percentage=rule["PERCENTAGE"],
                                 regex=rule["REGEX"],circuit=circuit)
    data_circuit_pairs.append((data,circuit,baseline_circuit))

MODEL_DIR = "model/tunning_models"
EVALUATION_DIR = "data/evaluation_data"
RESULTS_DIR = "results"
SPLIT_MANIFEST_PATH = "scaled_family_input_split_manifest.json"

# Runtime filters
FAMILY_FILTER = os.environ.get("FAMILY_FILTER", "").strip()
BACKEND_FILTER = os.environ.get("BACKEND_FILTER", "").strip()
MAX_BACKENDS = int(os.environ.get("MAX_BACKENDS", "0") or "0")

# Runtime controls
RUN_SHOTS = int(os.environ.get("RUN_SHOTS", "128") or "128")

# Checkpoint / resume
PROGRESS_DIR = "data/evaluation_progress"


def HellingerDistance(p, q):
    n = len(p)
    total = 0.0
    for i in range(n):
        total += (np.sqrt(p[i]) - np.sqrt(q[i])) ** 2
    return (1.0 / np.sqrt(2.0)) * np.sqrt(total)


def set_seed(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def deserialize_item(x):
    if isinstance(x, list):
        return tuple(deserialize_item(v) for v in x)
    return x


def load_manifest(path=SPLIT_MANIFEST_PATH):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for family_name, info in raw["families"].items():
        info["train_inputs"] = [deserialize_item(x) for x in info["train_inputs"]]
        info["eval_inputs"] = [deserialize_item(x) for x in info["eval_inputs"]]
    return raw


def filtered_backends():
    selected = backends
    if BACKEND_FILTER:
        selected = [x for x in selected if x[0] == BACKEND_FILTER]
    if MAX_BACKENDS > 0:
        selected = selected[:MAX_BACKENDS]
    return selected


def ensure_dirs(selected_backends):
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    for bk, _ in selected_backends:
        os.makedirs(os.path.join(EVALUATION_DIR, bk), exist_ok=True)
        # os.makedirs(os.path.join("..", "data", EVALUATION_DIR, bk), exist_ok=True)


def csv_output_path(backend_name, family_name):
    return os.path.join(EVALUATION_DIR, f"{backend_name}_{family_name}.csv")


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
        circuit.get_result(backend_ideal, inp, number_of_runs=RUN_SHOTS)
        with open(
            os.path.join(qasm_dir, f"{circuit.key_aurguments['ID']}_{iteration}.qasm"),
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:
            f.write(circuit.key_aurguments["circuit"].qasm())
    except MissingOptionalLibraryError as ex:
        print(ex)

    ideal = circuit.get_result(backend_ideal, inp, number_of_runs=RUN_SHOTS)
    noise = circuit.get_result(backend_noise, inp, number_of_runs=RUN_SHOTS)

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
    return rows


def process_backend_family(circuit, family_name, backend_ideal, backend_noise, inputs, backend_name):
    out_csv = csv_output_path(backend_name, family_name)
    prog = load_progress(backend_name, family_name)
    start_iter = int(prog.get("completed_iterations", 0))

    ensure_csv_header(out_csv)

    if start_iter >= len(inputs):
        print(f"[skip] {family_name} on {backend_name}: already completed ({start_iter}/{len(inputs)})")
        return

    qasm_dir = os.path.join(EVALUATION_DIR, backend_name)
    os.makedirs(qasm_dir, exist_ok=True)

    print(f"[resume] {family_name} on {backend_name}: start from iteration {start_iter}/{len(inputs)}")
    for iteration in tqdm(range(start_iter, len(inputs))):
        inp = inputs[iteration]
        try:
            rows = rows_for_one_iteration(circuit, backend_ideal, backend_noise, inp, iteration, qasm_dir)
            append_rows(out_csv, rows)
        except Exception as ex:
            print(f"[iteration-skip] family={family_name} backend={backend_name} iteration={iteration} error={ex}")
        finally:
            save_progress(backend_name, family_name, iteration + 1)


def main():
    if not os.path.exists(SPLIT_MANIFEST_PATH):
        raise FileNotFoundError(
            f"Missing split manifest: {SPLIT_MANIFEST_PATH}. Run BaselineTuner.py first."
        )

    split_payload = load_manifest(SPLIT_MANIFEST_PATH)
    selected_backends = filtered_backends()
    ensure_dirs(selected_backends)

    print("CUTs:", split_payload["CUTs"])
    print("Evaluation counts by family:", split_payload["eval_counts"])
    print("Evaluation total:", split_payload["actual_eval_total"])
    print("Backends to run:", [bk for bk, _ in selected_backends])

    backend_factory = BackendFactory()
    backend_ideal = backend_factory.initialize_backend()
    backend_executors = {bk: backend_factory.initialize_backend(bk) for bk, _ in selected_backends}

    for bk, _ in selected_backends:
        print(f"Generating Data For {bk} Backend")
        print("------------------------------------------")
        for family_name in split_payload["CUTs"]:
            if FAMILY_FILTER and family_name != FAMILY_FILTER:
                continue

            eval_inputs = split_payload["families"][family_name]["eval_inputs"]
            print(f"Executing family={family_name} | eval circuits={len(eval_inputs)}")
            circuit = get_circuit_class_object(family_name)
            process_backend_family(
                circuit,
                family_name,
                backend_ideal,
                backend_executors[bk],
                eval_inputs,
                bk,
            )

    # Result evaluation intentionally left disabled here.


# if __name__ == "__main__":
#     main()
for seed in SEEDS:
    print("============================================================")
    print("Current seed: ", seed)
    print("============================================================")
    set_seed(seed)
    RQ1C = defaultdict(lambda:{})

    os.makedirs(f"results/seed_{seed}", exist_ok=True)
    for bk, qubit_size in tqdm(backends):
        print("Generating Result For {} Backend".format(bk))
        print("------------------------------------------")
        # RQ1C[bk] = {}

        for data,circuit,name in data_circuit_pairs:
            print("---------------------{}---------------------".format(name))
            RQ1C.setdefault(bk, {}).setdefault(name, [])
            RQ1C[bk][name] = []

            data_frame = pd.read_csv("data/evaluation_data/{}/{}_{}.csv".format(bk,bk,name),dtype={'circuit': 'string'}).dropna()
            data_frame["circuit"] = data_frame["circuit"].astype(str)
            
            predictor = ktrain.load_predictor('model/tunning_models/seed_{}/{}_{}'.format(seed,bk,name))
            table_by_cirid = {
                cid: sub[["POF","ODR","POS","Target Value"]].to_dict("records")
                for cid, sub in data_frame.groupby("circuit")
            }

            # filter_probs = []
            # ideal_probs = []
            for key, value in table_by_cirid.items():
                print(key, value)
                filter_probs = []
                ideal_probs = []
                for v in value:
                    pof = v["POF"]
                    odr = v["ODR"]
                    pos = v["POS"]
                    temp = pd.DataFrame([[pof,odr,pos]],columns=["POF","ODR","POS"])

                    prediction = predictor.predict(temp)[0]
                    #print(prediction)
                    if prediction[0]<0:
                        filter_probs.append(0)
                    elif prediction[0]>1:
                        filter_probs.append(1)
                    else:
                        filter_probs.append(prediction[0])
                    ideal_probs.append(v["Target Value"])
            
                PF = np.array(ideal_probs).reshape(-1,1)
                QF = np.array(filter_probs).reshape(-1,1)
            # print(PF,QF,ideal_probs)
                HL_filter = HellingerDistance(PF,QF)
            # Filter_TVD.append(TVD_filter)
            # JHN_filter = JHN(PF,QF)[0]
            # Filter_JHN.append(JHN_filter)
            # HL_filter = HellingerDistance(PF,QF)[0]
            # Filter_HL.append(HL_filter)
        
            # RQ1C[bk][name] = {"FilterTVD":Filter_TVD,
            #                 "FilterJHN":Filter_JHN,
            #                 "FilterHL":Filter_HL,
            #                 }
                RQ1C[bk][name].append(HL_filter)

            
            saveRQ1 = {k: v for k, v in RQ1C.items()}
            rqfile = open(f"results/seed_{seed}/saveRQ1.json","wb")
            pickle.dump(saveRQ1,rqfile)
            rqfile.close()