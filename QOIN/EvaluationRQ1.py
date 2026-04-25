import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

import urllib.request
import pandas as pd
import numpy as np
import time
# import ktrain
# from ktrain import tabular
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
import ktrain
from ktrain import tabular
import json
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from qiskit.exceptions import MissingOptionalLibraryError
warnings.filterwarnings('ignore')

import os
import pickle
import tensorflow as tf


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


from scipy.spatial.distance import jensenshannon as JHN

def TVD(p, q):
    return 0.5 * np.abs(p - q).sum()


def HellingerDistance(p, q):
    n = len(p)
    sum_ = 0.0
    for i in range(n):
        sum_ += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum_)
    return result


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


backends = [('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65), ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28), ('FakeCasablanca', 7), ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJakarta', 7), ('FakeJohannesburg', 20), ('FakeKolkata', 27), ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27), ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27), ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27), ('FakeToronto', 27), ('FakeWashington', 127)]
BaselineCircuits,CUTs = train_test_split(get_all_circuits(),train_size=0.4,random_state=13)
SEEDS = [1, 2, 3, 4, 5]

# # Get Evaluation data for CUTs



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

#data_circuit_pairs


# # Execute on Backends


backend_factory = BackendFactory()
backend = backend_factory.initialize_backend()
backend_executor = {}
for bk, qubit_size in tqdm(backends):
    backend_executor[bk] = backend_factory.initialize_backend(bk)

os.makedirs(f"evaluation_data", exist_ok=True)
os.makedirs(f"results", exist_ok=True)
RQ1C = defaultdict(lambda:{})

for bk, qubit_size in tqdm(backends):
    os.makedirs(f"../data/evaluation_data/{bk}", exist_ok=True)
    print("Generating Data For {} Backend".format(bk))
    print("------------------------------------------")
    # RQ1C[bk] = {}

    for data,circuit,name in data_circuit_pairs:
        RQ1C.setdefault(bk, {}).setdefault(name, [])
        RQ1C[bk][name] = []
        data_rows = []
        single_row = []
        print("Executing CUT circuit ID: {}".format(circuit.key_aurguments["ID"]), "input_type:", circuit.key_aurguments["input_type"])

        if circuit.key_aurguments["input_type"]==1:
            iteration = 0

            for inp in data:

                try:
                    ideal = circuit.get_result(backend,inp)
                    # print(circ.qasm())
                    # print(circuit.key_aurguments["circuit"].qasm())
                    print(str(circuit.key_aurguments["ID"]) + "_" + str(iteration))
                    with open(f"../data/evaluation_data/{bk}/{str(circuit.key_aurguments['ID'])}_{str(iteration)}.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(circuit.key_aurguments["circuit"].qasm())
                except MissingOptionalLibraryError as ex:
                    print(ex)
                
                ideal = circuit.get_result(backend,inp)
                Noise = circuit.get_result(backend_executor[bk],inp)
                # noise_data.append({"{}".format(inp):Noise})
                # ideal_data.append({"{}".format(inp):ideal})
                # print(ideal["probability"])
                # print(Noise["probability"])
                for outputs in ideal["probability"]:
                    target_variable_prob = None
                    target_variable_odds = None
                    actual_target_prob = outputs["prob"]
                    all_other_probs = 0
                    for noise_outputs in Noise["probability"]:
                        if outputs["bin"] == noise_outputs["bin"]:
                            target_variable_prob = noise_outputs["prob"]
                            target_variable_odds = noise_outputs["odds"]
                            print(outputs["bin"], target_variable_prob, actual_target_prob)
                        else:
                            all_other_probs += noise_outputs["prob"]
                        
                    temp_row = [x for x in single_row]
                    temp_row.extend([all_other_probs,target_variable_odds,target_variable_prob,actual_target_prob, str(circuit.key_aurguments["ID"]) + "_" + str(iteration)])
                    data_rows.append(temp_row)
            
                iteration += 1

        elif circuit.key_aurguments["input_type"]==2:
            pairs = [[x,y] for y in data for x in data]
            iteration = 0

            for inp in pairs:
                
                try:
                    ideal = circuit.get_result(backend,inp)
                    # print(circ.qasm())
                    # print(circuit.key_aurguments["circuit"].qasm())
                    print(str(circuit.key_aurguments["ID"]) + "_" + str(iteration))
                    with open(f"../data/evaluation_data/{bk}/{str(circuit.key_aurguments['ID'])}_{str(iteration)}.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(circuit.key_aurguments["circuit"].qasm())
                except MissingOptionalLibraryError as ex:
                    print(ex)
                
                ideal = circuit.get_result(backend,inp)
                Noise = circuit.get_result(backend_executor[bk],inp)
                # noise_data.append({"{}:{}".format(inp[0],inp[1]):Noise})
                # ideal_data.append({"{}:{}".format(inp[0],inp[1]):ideal})
                # print(ideal["probability"])
                # print(Noise["probability"])
                for outputs in ideal["probability"]:
                    target_variable_prob = None
                    target_variable_odds = None
                    actual_target_prob = outputs["prob"]
                    all_other_probs = 0
                    for noise_outputs in Noise["probability"]:
                        if outputs["bin"] == noise_outputs["bin"]:
                            target_variable_prob = noise_outputs["prob"]
                            target_variable_odds = noise_outputs["odds"]
                            print(outputs["bin"], target_variable_prob, actual_target_prob)
                        else:
                            all_other_probs += noise_outputs["prob"]
                    temp_row = [x for x in single_row]
                    temp_row.extend([all_other_probs,target_variable_odds,target_variable_prob,actual_target_prob, str(circuit.key_aurguments["ID"]) + "_" + str(iteration)])
                    data_rows.append(temp_row)
            
                iteration += 1

    # #-=-=-=-=-=-=-==-=-=-=-=-=-=-==-Saving json-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        
        columns = []
        columns.extend(["POF","ODR","POS","Target Value","circuit"])
        data_frame = pd.DataFrame(data_rows,columns=columns)
        data_frame.to_csv("../data/evaluation_data/{}_{}.csv".format(bk,name),index=False)
        data_frame.to_csv("evaluation_data/{}_{}.csv".format(bk,name),index=False)

    # #-=-=-=-=-=-=-==-=-=-=-=-=-=-==-Evaluation-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
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

            data_frame = pd.read_csv("evaluation_data/{}_{}.csv".format(bk,name),dtype={'circuit': 'string'}).dropna()
            data_frame["circuit"] = data_frame["circuit"].astype(str)
            
            predictor = ktrain.load_predictor('tunning_models/seed_{}/{}_{}'.format(seed,bk,name))
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

            
            

