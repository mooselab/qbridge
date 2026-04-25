import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

import urllib.request
import pandas as pd
import numpy as np
import time
import ktrain
from ktrain import tabular
from sklearn.model_selection import train_test_split
from tqdm import *
pd.set_option('display.max_columns', None)

from benchmark_circuits import *
import random
import pandas as pd
from tqdm import *
import pkgutil
import warnings
import exrex
import math
import time
import json
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from statsmodels.stats.gof import chisquare
import pickle
warnings.filterwarnings('ignore')

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import ktrain
import tensorflow as tf
from benchmark_circuits import *
import rpy2.robjects as robjects
r = robjects.r
r['source']('chisquare.R')

# ----------------- CONFIG -----------------
DEBUG_PRINT_EVERY = 0
MAX_INPUTS_PER_CUT = None
PARTIAL_SAVE_EVERY = 1
OUT_TXT = "results/qoin.txt"
# OUT_TXT = "results/no_mitigation.txt"
OUT_PICKLE = None
# ------------------------------------------

def Uof(observed, expected):
    if not observed:
        return "F"
    for k in observed.keys():
        if k not in expected:
            return "F"
    return "P"

def Wodf(observed, expected, threshold=0.01):
    """R chisquare: p<0.01 return F, else return P"""
    test = robjects.globalenv['chisquare']
    try:
        if len(observed) == 1 and len(expected) == 1:
            return "P"
        keys = sorted(set(expected.keys()) | set(observed.keys()))
        exp = [expected.get(k, 0.0) for k in keys]
        obs = [observed.get(k, 0.0) for k in keys]
        p = float(np.array(test(robjects.FloatVector(obs), robjects.FloatVector(exp)))[0])
        return "F" if p < threshold else "P"
    except Exception:
        return "F"

def convertNaQp2QuCAT_notation(output, value="prob"):
    # output["probability"] includes {"bin": "...", value: ...}
    return {x["bin"]: x[value] for x in output.get("probability", [])}

def _predict_probs_vec(prob_rows, predictor):
    """
    prob_rows: list of dicts, every dict at least contains {"prob": float, "odds": float}
    """
    if not prob_rows:
        return np.array([], dtype=float)
    pos = np.array([row["prob"] for row in prob_rows], dtype=float)
    odr = np.array([row["odds"] for row in prob_rows], dtype=float)
    pof = 1.0 - pos
    df = pd.DataFrame({"POF": pof, "ODR": odr, "POS": pos}, columns=["POF", "ODR", "POS"])
    preds = predictor.predict(df)
    preds = np.asarray(preds).reshape(-1)
    return np.clip(preds, 0.0, 1.0)

def filter_output_fast(output, predictor, count=True, shots=1024):
    """
        A faster and more memory-efficient version:
        - Feed all states into the model at once as a DataFrame
        - After prediction, clip outputs to [0,1]
        - If count=True, convert results to integer counts based on shots; otherwise, return probabilities directly
        Returns: (filtered_output: dict[str, int | float], prediction_output: dict[str, [count, pred]])
    """

    prob_rows = output.get("probability", [])
    if not prob_rows:
        return {}, {}

    bins = [row["bin"] for row in prob_rows]
    counts = [row.get("count", 0) for row in prob_rows]

    preds = _predict_probs_vec(prob_rows, predictor)

    prediction_output = {b: [c, float(p)] for b, c, p in zip(bins, counts, preds)}

    if count:
        filtered = {b: int(round(p * shots)) for b, p in zip(bins, preds) if p > 0.0}
    else:
        filtered = {b: float(p) for b, p in zip(bins, preds) if p > 0.0}

    return filtered, prediction_output

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

backends = [('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65), ('FakeCairo', 27),
            ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28), ('FakeCasablanca', 7),
            ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJakarta', 7), ('FakeJohannesburg', 20),
            ('FakeKolkata', 27), ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27),
            ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27), ('FakeRochester', 53),
            ('FakeSingapore', 20), ('FakeSydney', 27), ('FakeToronto', 27), ('FakeWashington', 127)]
SEEDS = [1, 2, 3, 4, 5]

BaselineCircuits, CUTs = train_test_split(get_all_circuits(), train_size=0.4, random_state=13)

with open('results/Official_result.pickle', 'rb') as f:
    Official_result = pickle.load(f)

with open('results/Mutation_result.pickle', 'rb') as f:
    Mutation_result = pickle.load(f)

def process_one_case(ps_spec, noisy_out, predictor, official_set, counters):
    """
    ps_spec: dict, expected distribution (probabilities)
    noisy_out: raw noisy output (includes probability list)
    official_set: the Negative list from Official_result[bk][cut]['origin'/'mutantX']
    counters: mutable dict of (TP, FP, TN, FN)
    Returns: whether it is judged as F (used to accumulate fail scores for the cut)
    """
    ps = convertNaQp2QuCAT_notation(ps_spec, value='prob')
    if not predictor:
        filtered = convertNaQp2QuCAT_notation(noisy_out,value='count')
    else:
        filtered, _pred = filter_output_fast(noisy_out, predictor, count=False)

    res = Uof(filtered, ps)
    if res == "P":
        res = Wodf(filtered, ps)

    is_fail = (res == "F")

    if is_fail:
        if counters["idx"] in official_set:
            counters["TN"] += 1
        else:
            counters["FN"] += 1
    else:
        if counters["idx"] in official_set:
            counters["FP"] += 1
        else:
            counters["TP"] += 1
    return is_fail



for seed in SEEDS:
    print("============================================================")
    print("Current seed: ", seed)
    print("============================================================")
    set_seed(seed)
    RQ2C = {}
    global_counters = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "TOTAL": 0}
    OUT_TXT = f'results/qoin_seed{seed}.txt'

    for bk, _ in tqdm(backends, desc="Backends"):
        backend_result = []
        for cut in tqdm(CUTs, desc=f"{bk} CUTs", leave=False):
            # for qoin
            try:
                predictor = ktrain.load_predictor(f'tunning_models/seed_{seed}/{bk}_{cut}')
            except Exception as e:
                print(f"[WARN] load_predictor failed for {bk}-{cut}: {e}")
                continue
            # for no mitigation
            # predictor = None

            algo = get_circuit_class_object(cut)
            inputs = algo.get_full_inputs()
            if MAX_INPUTS_PER_CUT is not None:
                inputs = inputs[:MAX_INPUTS_PER_CUT]

            original_fail = mutant1_fail = mutant2_fail = mutant3_fail = 0

            for idx, inp in enumerate(inputs, start=1):
                
                try:
                    rec = Mutation_result[bk][cut][idx]
                except Exception as e:
                    if DEBUG_PRINT_EVERY and idx % DEBUG_PRINT_EVERY == 0:
                        print(f"[SKIP rec-missing] {bk} {cut} idx={idx}: {e}")
                    continue

                # ---- origin ----
                counters = {"TP":0,"FP":0,"TN":0,"FN":0,"idx": idx}
                is_fail = process_one_case(
                    rec['origin']['ps'],
                    rec['origin']['ps_noise'],
                    predictor,
                    set(Official_result[bk][cut]['origin']),
                    counters
                )
                original_fail += int(is_fail)

                # ---- mutant1 ----
                counters["idx"] = idx
                is_fail = process_one_case(
                    rec['mutant1']['ps'],
                    rec['mutant1']['ps_noise'],
                    predictor,
                    set(Official_result[bk][cut]['mutant1']),
                    counters
                )
                mutant1_fail += int(is_fail)

                # ---- mutant2 ----
                counters["idx"] = idx
                is_fail = process_one_case(
                    rec['mutant2']['ps'],
                    rec['mutant2']['ps_noise'],
                    predictor,
                    set(Official_result[bk][cut]['mutant2']),
                    counters
                )
                mutant2_fail += int(is_fail)

                # ---- mutant3 ----
                counters["idx"] = idx
                is_fail = process_one_case(
                    rec['mutant3']['ps'],
                    rec['mutant3']['ps_noise'],
                    predictor,
                    set(Official_result[bk][cut]['mutant3']),
                    counters
                )
                mutant3_fail += int(is_fail)
                
                global_counters["TP"] += counters["TP"]
                global_counters["FP"] += counters["FP"]
                global_counters["TN"] += counters["TN"]
                global_counters["FN"] += counters["FN"]
                global_counters["TOTAL"] += 4

                if DEBUG_PRINT_EVERY and idx % DEBUG_PRINT_EVERY == 0:
                    print(f"[{bk} {cut}] idx={idx} TP={global_counters['TP']} FP={global_counters['FP']} TN={global_counters['TN']} FN={global_counters['FN']}")

                if idx % 200 == 0:
                    gc.collect()

            n = max(1, len(inputs))
            backend_result.append({
                cut: {
                    "original_score": 100.0 * original_fail / n,
                    "mutant1_score":  100.0 * mutant1_fail  / n,
                    "mutant2_score":  100.0 * mutant2_fail  / n,
                    "mutant3_score":  100.0 * mutant3_fail  / n,
                }
            })

        RQ2C[bk] = backend_result

        if PARTIAL_SAVE_EVERY and (len(RQ2C) % PARTIAL_SAVE_EVERY == 0):
            with open(OUT_TXT, "w") as f:
                f.write(f"TP = {global_counters['TP']}\n")
                f.write(f"FP = {global_counters['FP']}\n")
                f.write(f"TN = {global_counters['TN']}\n")
                f.write(f"FN = {global_counters['FN']}\n")
                f.write(f"TOTAL = {global_counters['TOTAL']}\n")
            if OUT_PICKLE:
                with open(OUT_PICKLE, "wb") as f:
                    pickle.dump(RQ2C, f)

    with open(OUT_TXT, "w") as f:
        f.write(f"TP = {global_counters['TP']}\n")
        f.write(f"FP = {global_counters['FP']}\n")
        f.write(f"TN = {global_counters['TN']}\n")
        f.write(f"FN = {global_counters['FN']}\n")
        f.write(f"TOTAL = {global_counters['TOTAL']}\n")

    if OUT_PICKLE:
        with open(OUT_PICKLE, "wb") as f:
            pickle.dump(RQ2C, f)



