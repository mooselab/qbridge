from benchmark_circuits import *
import random
import pandas as pd
import numpy as np
from tqdm import *
import pkgutil
import warnings
import exrex
import math
import time
from qiskit.exceptions import MissingOptionalLibraryError
warnings.filterwarnings('ignore')

import os



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


# # Select Baseline Circuits


from sklearn.model_selection import train_test_split
backends = [('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65), ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28), ('FakeCasablanca', 7), ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJakarta', 7), ('FakeJohannesburg', 20), ('FakeKolkata', 27), ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27), ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27), ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27), ('FakeToronto', 27), ('FakeWashington', 127)]
BaselineCircuits,CUTs = train_test_split(get_all_circuits(),train_size=0.4,random_state=13)


# # Generate Data

rules = read_configuration("Configuration.txt")
data_circuit_pairs = []
for baseline_circuit in BaselineCircuits:
    circuit = get_circuit_class_object(baseline_circuit)
    print("Generating data for baseline circuit: {}".format(baseline_circuit))
    rule = rules[circuit.key_aurguments["ID"]]
    data,circuit = generate_data(Format=rule["FORMAT"],startRange=rule["START"],
                                 endRange=rule["END"],percentage=rule["PERCENTAGE"],
                                 regex=rule["REGEX"],circuit=circuit)
    data_circuit_pairs.append((data,circuit))

# print(data_circuit_pairs)


# # Execute on noisy and ideal backend


backend_factory = BackendFactory()
backend = backend_factory.initialize_backend()
backend_executors = {}
print("initializing backends")
for bk, qubit_size in tqdm(backends):
    backend_executors[bk] = backend_factory.initialize_backend(bk)
os.makedirs(f"baseline_training_data", exist_ok=True)
for bk, qubit_size in backends:
    os.makedirs(f"../data/baseline_training_data/{bk}", exist_ok=True)
    data_rows = []
    single_row = []

    print("Generating Data For {} Backend".format(bk))
    print("------------------------------------------")

    for data,circuit in data_circuit_pairs:
        print(circuit.key_aurguments["ID"], data, circuit, '\n')

        print("Executing baseline circuit ID: {}".format(circuit.key_aurguments["ID"]))

        if circuit.key_aurguments["input_type"]==1:
            for iteration in tqdm(range(np.power(len(data),2))):
                inp = data[iteration%len(data)]
                try:
                    ideal = circuit.get_result(backend,inp)
                    # print(circ.qasm())
                    # print(circuit.key_aurguments["circuit"].qasm())
                    print(str(circuit.key_aurguments["ID"]) + "_" + str(iteration))
                    with open(f"../data/baseline_training_data/{bk}/{str(circuit.key_aurguments['ID'])}_{str(iteration)}.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(circuit.key_aurguments["circuit"].qasm())
                except MissingOptionalLibraryError as ex:
                    print(ex)
                # continue
                # print(inp)
                # continue
                try:
                    ideal = circuit.get_result(backend,inp)
                    Noise = circuit.get_result(backend_executors[bk],inp)
                except MissingOptionalLibraryError as ex:
                    print(ex)
                    continue

                for outputs in ideal["probability"]:
                    target_variable_prob = None
                    target_variable_odds = None
                    actual_target_prob = outputs["prob"]
                    all_other_probs = 0
                    for noise_outputs in Noise["probability"]:
                        if outputs["bin"] == noise_outputs["bin"]:
                            target_variable_prob = noise_outputs["prob"]
                            target_variable_odds = noise_outputs["odds"]
                        else:
                            all_other_probs += noise_outputs["prob"]
                    temp_row = [x for x in single_row]
                    temp_row.extend([all_other_probs,target_variable_odds,target_variable_prob,actual_target_prob, str(circuit.key_aurguments["ID"]) + "_" + str(iteration)])
                    data_rows.append(temp_row)

        elif circuit.key_aurguments["input_type"]==2:
            pairs = [[x,y] for y in data for x in data]
            for inp in tqdm(pairs):
                try:
                    ideal = circuit.get_result(backend,inp)
                    # print(circ.qasm())
                    # print(circuit.key_aurguments["circuit"].qasm())
                    print(str(circuit.key_aurguments["ID"]) + "_" + str(iteration))
                    with open(f"../data/baseline_training_data/{bk}/{str(circuit.key_aurguments['ID'])}_{str(iteration)}.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(circuit.key_aurguments["circuit"].qasm())
                except MissingOptionalLibraryError as ex:
                    print(ex)
                # continue
                # print(inp)
                # continue
                ideal = circuit.get_result(backend,inp)
                Noise = circuit.get_result(backend_executors[bk],inp)

                for outputs in ideal["probability"]:
                    target_variable_prob = None
                    target_variable_odds = None
                    actual_target_prob = outputs["prob"]
                    all_other_probs = 0
                    for noise_outputs in Noise["probability"]:
                        if outputs["bin"] == noise_outputs["bin"]:
                            target_variable_prob = noise_outputs["prob"]
                            target_variable_odds = noise_outputs["odds"]
                        else:
                            all_other_probs += noise_outputs["prob"]
                    temp_row = [x for x in single_row]
                    temp_row.extend([all_other_probs,target_variable_odds,target_variable_prob,actual_target_prob, str(circuit.key_aurguments["ID"]) + "_" + str(iteration)])
                    data_rows.append(temp_row)

    # #-=-=-=-=-=-=-==-=-=-=-=-=-=-==-Appending to CSV-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    columns = []
    columns.extend(["POF","ODR","POS","Target Value","circuit"])
    data_frame = pd.DataFrame(data_rows,columns=columns)
    data_frame.to_csv("../data/baseline_training_data/{}/{}.csv".format(bk,bk),index=False)
    data_frame.to_csv("baseline_training_data/{}.csv".format(bk),index=False)




