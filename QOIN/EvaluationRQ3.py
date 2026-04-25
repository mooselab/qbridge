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



backends = [('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65), ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28), ('FakeCasablanca', 7), ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJakarta', 7), ('FakeJohannesburg', 20), ('FakeKolkata', 27), ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27), ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27), ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27), ('FakeToronto', 27), ('FakeWashington', 127)]
BaselineCircuits,CUTs = train_test_split(get_all_circuits(),train_size=0.4,random_state=13)


import rpy2.robjects as robjects
r = robjects.r
r['source']('chisquare.R')

def Uof(observed,expected):
    if len(observed.keys())<1:
        return "F"
    for k in observed.keys():
        if k not in expected.keys():
            return "F"
    return "P"
    
def Wodf(observed,expected):
    test = robjects.globalenv['chisquare']
    try:
        if len(observed)==1 and len(expected)==1:
            return "P"
        
        obs = []
        exp = []
        expected = dict(sorted(expected.items(), key=lambda item: item[0]))
        observed = dict(sorted(observed.items(), key=lambda item: item[0]))
        for k in set(observed.keys()).intersection(expected.keys()):
            obs.append(observed[k])
            exp.append(expected[k])
        
        for k in set(expected.keys()).difference(observed.keys()):
            exp.append(expected[k])
        
        if len(obs)<len(exp):
#             epsilon = 1024-sum(obs)
#             try:
#                 epsilon = epsilon/(len(exp)-len(obs))
#             except:
#                 epsilon = 0
            obs.extend([0 for t in range(len(exp)-len(obs))])
        
        #obs = [int(o*100) for o in obs]
        
        df_result_r = test(robjects.FloatVector(obs),robjects.FloatVector(exp))
        p = np.array(df_result_r)[0]
#         print("expected_o:",expected)
#         print("observed_o:",observed)
#         print("expected:",exp)
#         print("observed:",obs)
#         print("p-value",p)
        if p<0.01:
            return "F"
        else:
            return "P"
    except Exception as e:
        print(e)
        return "F"
    
def convertNaQp2QuCAT_notation(output,value="prob"):
    program_specification = {}
    for x in output["probability"]:
        program_specification[x["bin"]] = x[value]
    return program_specification

def filter_output(output,predictor,count=True):
    prediction_output = {}
    for state in output["probability"]:
        all_other_probs = sum([x["prob"] for x in output["probability"] if x["bin"]!=state["bin"]])
        odr = state["odds"]
        pos = state["prob"]
        pof = 1-pos
        df = pd.DataFrame([[pof,odr,pos]],columns=["POF","ODR","POS"])
        prediction = predictor.predict(df)[0][0]
        prediction_output[state["bin"]] = [state["count"],prediction]
        
    filtered_output = {}
    # clamp output to 0 and 1
    for k in prediction_output.keys():
        if prediction_output[k][1]>1:
            if np.abs(prediction_output[k][1]-1)<0.2:
                prediction_output[k][1] = min(prediction_output[k][1],1) 
                filtered_output[k] = prediction_output[k]
            else:
                if prediction_output[k][1]>2:
                    prediction_output[k][1] = prediction_output[k][1]%1
                    filtered_output[k] = prediction_output[k]
                
        elif prediction_output[k][1]<0:
            prediction_output[k][1] = 0.0
            filtered_output[k] = prediction_output[k]
        else:
            filtered_output[k] = prediction_output[k]

    # check for irrelevent values
#     if len(filtered_output.keys())>2:
#         temp = {}
#         maxvalue = max([filtered_output[k][0] for k in filtered_output.keys()])
#         for k in filtered_output.keys():
#             if (filtered_output[k][0]/maxvalue)>=0.35:
#                 temp[k] = filtered_output[k]
#         filtered_output =  temp
    
    temp1 = {}
    total = sum([filtered_output[k][0] for k in filtered_output.keys()])
    for k in filtered_output.keys():
        if filtered_output[k][1]==0:
            continue
        if count:
            temp1[k] = int(filtered_output[k][1]*1024)#int(filtered_output[k][1]*total)
        else:
            temp1[k] = filtered_output[k][1]
    filtered_output =  temp1
        
            
    return filtered_output,prediction_output


def filter_output_all(outputs,predictor,count=True):
    results = []
    for output in outputs:
        prediction_output = {}
        df = pd.DataFrame(columns=["POF","ODR","POS"])
        for state in output["probability"]:
            all_other_probs = sum([x["prob"] for x in output["probability"] if x["bin"]!=state["bin"]])
            odr = state["odds"]
            pos = state["prob"]
            pof = 1-pos
            df = df.append({"POF":pof,"ODR":odr,"POS":pos},ignore_index=True)
        
        predictions = predictor.predict(df)
        
        for state,prediction in zip(output["probability"],predictions):
            prediction_output[state["bin"]] = [state["count"],prediction[0]]
        
        #print(predictions,prediction_output)
        
        filtered_output = {}
        # clamp output to 0 and 1
        for k in prediction_output.keys():
            if prediction_output[k][1]>1:
                if np.abs(prediction_output[k][1]-1)<0.2:
                    prediction_output[k][1] = min(prediction_output[k][1],1) 
                    filtered_output[k] = prediction_output[k]
                else:
                    if prediction_output[k][1]>2:
                        prediction_output[k][1] = prediction_output[k][1]%1
                        filtered_output[k] = prediction_output[k]

            elif prediction_output[k][1]<0:
                prediction_output[k][1] = 0.0
                filtered_output[k] = prediction_output[k]
            else:
                filtered_output[k] = prediction_output[k]

    
        temp1 = {}
        total = sum([filtered_output[k][0] for k in filtered_output.keys()])
        for k in filtered_output.keys():
            if filtered_output[k][1]==0:
                continue
            if count:
                temp1[k] = int(filtered_output[k][1]*1024)#int(filtered_output[k][1]*total)
            else:
                temp1[k] = filtered_output[k][1]
        filtered_output =  temp1
        
        results.append((filtered_output,prediction_output))
    
    return results


def default_inner_template():
    return {"uof":[],"uof_w":[],"wodf":[],"wodf_w":[]}



backend_factory = BackendFactory()
backend = backend_factory.initialize_backend()
backend_executors = {}
for bk,_ in tqdm(backends):
    backend_executors[bk] = backend_factory.initialize_backend(name=bk)



Official_result = {}
uof_dict = {}
for bk,_ in tqdm(backends):
    backend_noise = backend_executors[bk]
    backend_result = []
    print(bk)

    for cut in CUTs:
        print(cut)
        algo = get_circuit_class_object(cut)
        total_inputs = 0
        
        Official_result.setdefault(bk, {}).setdefault(cut,{'origin': [], 'mutant1': [], 'mutant2': [], 'mutant3': []})
        uof_dict.setdefault(bk, {}).setdefault(cut,{'origin': [], 'mutant1': [], 'mutant2': [], 'mutant3': []})
        for inp in algo.get_full_inputs():

            total_inputs+=1

            ps = algo.get_result(backend,inp)
            ps_count = algo.get_result(backend,inp)
            ps = convertNaQp2QuCAT_notation(ps,value='prob')
            ps_count = convertNaQp2QuCAT_notation(ps_count,value='count')

            result = Uof(ps_count,ps)
            if result=="P":
                result = Wodf(ps_count,ps)
                if result=="F":
                    Official_result[bk][cut]['origin'].append(total_inputs)
            else:
                Official_result[bk][cut]['origin'].append(total_inputs)
                uof_dict[bk][cut]['origin'].append(total_inputs)

            # mutant 1:
            mutant1 = get_circuit_class_object_mutation(cut+"_M1")
            mutant_output = mutant1.get_result(backend,inp)
            mutant_output = convertNaQp2QuCAT_notation(mutant_output,value='count')

            result = Uof(mutant_output,ps)
            if result=="P":
                result = Wodf(mutant_output,ps)
                if result=="F":
                    Official_result[bk][cut]['mutant1'].append(total_inputs)
            else:
                Official_result[bk][cut]['mutant1'].append(total_inputs)
                uof_dict[bk][cut]['mutant1'].append(total_inputs)
                
            # mutant 2:
            mutant2 = get_circuit_class_object_mutation(cut+"_M2")
            mutant_output = mutant2.get_result(backend,inp)
            mutant_output = convertNaQp2QuCAT_notation(mutant_output,value='count')

            result = Uof(mutant_output,ps)
            if result=="P":
                result = Wodf(mutant_output,ps)
                if result=="F":
                    Official_result[bk][cut]['mutant2'].append(total_inputs)
            else:
                Official_result[bk][cut]['mutant2'].append(total_inputs)
                uof_dict[bk][cut]['mutant2'].append(total_inputs)

            # mutant 1:
            mutant3 = get_circuit_class_object_mutation(cut+"_M3")
            mutant_output = mutant3.get_result(backend,inp)
            mutant_output = convertNaQp2QuCAT_notation(mutant_output,value='count')
            
            result = Uof(mutant_output,ps)
            if result=="P":
                result = Wodf(mutant_output,ps)
                if result=="F":
                    Official_result[bk][cut]['mutant3'].append(total_inputs)
            else:
                Official_result[bk][cut]['mutant3'].append(total_inputs)
                uof_dict[bk][cut]['mutant3'].append(total_inputs)


file = open("results/Official_result.pickle","wb")
pickle.dump(Official_result,file)
file.close()

file = open("../data/testing_data/results/Official_result.pickle","wb")
pickle.dump(Official_result,file)
file.close()

file = open("../data/testing_data/results/UOF_result.pickle","wb")
pickle.dump(uof_dict,file)
file.close()



Mutation_result = {}
for bk,_ in tqdm(backends):
    backend_noise = backend_executors[bk]
    backend_result = []
    os.makedirs(f"../data/testing_data/{bk}", exist_ok=True)

    for cut in CUTs:
        algo = get_circuit_class_object(cut)
        total_inputs = 0
        for inp in algo.get_full_inputs():
            print(inp)
            total_inputs+=1
            Mutation_result.setdefault(bk, {}).setdefault(cut, {}).setdefault(total_inputs, {'origin': {}, 'mutant1': {}, 'mutant2': {}, 'mutant3': {}})

            ps = algo.get_result(backend,inp)
            ps_noise = algo.get_result(backend_noise,inp)
            with open(f"testing_data/{bk}/{str(algo.key_aurguments['ID'])}_{str(total_inputs)}_origin.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(algo.key_aurguments["circuit"].qasm())

            Mutation_result[bk][cut][total_inputs]['origin']['ps'] = ps
            Mutation_result[bk][cut][total_inputs]['origin']['ps_noise'] = ps_noise

            # mutant 1:
            mutant1 = get_circuit_class_object_mutation(cut+"_M1")
            mutant_output = mutant1.get_result(backend_noise,inp)
            with open(f"testing_data/{bk}/{str(mutant1.key_aurguments['ID'])}_{str(total_inputs)}_mutant1.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(mutant1.key_aurguments["circuit"].qasm())

            Mutation_result[bk][cut][total_inputs]['mutant1']['ps'] = ps
            Mutation_result[bk][cut][total_inputs]['mutant1']['ps_noise'] = mutant_output
                
            # mutant 2:
            mutant2 = get_circuit_class_object_mutation(cut+"_M2")
            mutant_output = mutant2.get_result(backend_noise,inp)

            with open(f"testing_data/{bk}/{str(mutant2.key_aurguments['ID'])}_{str(total_inputs)}_mutant2.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(mutant2.key_aurguments["circuit"].qasm())

            Mutation_result[bk][cut][total_inputs]['mutant2']['ps'] = ps
            Mutation_result[bk][cut][total_inputs]['mutant2']['ps_noise'] = mutant_output

            # mutant 3:
            mutant3 = get_circuit_class_object_mutation(cut+"_M3")
            mutant_output = mutant3.get_result(backend_noise,inp)

            with open(f"testing_data/{bk}/{str(mutant3.key_aurguments['ID'])}_{str(total_inputs)}_mutant3.qasm", "w", encoding="utf-8", newline="\n") as f:
                        f.write(mutant3.key_aurguments["circuit"].qasm())

            Mutation_result[bk][cut][total_inputs]['mutant3']['ps'] = ps
            Mutation_result[bk][cut][total_inputs]['mutant3']['ps_noise'] = mutant_output

    file = open("results/Mutation_result.pickle","wb")
    pickle.dump(Mutation_result,file)
    file.close()

    file = open("../data/testing_data/results/Mutation_result.pickle","wb")
    pickle.dump(Mutation_result,file)
    file.close()

file = open("results/Mutation_result.pickle","wb")
pickle.dump(Mutation_result,file)
file.close()

file = open("../data/testing_data/results/Mutation_result.pickle","wb")
pickle.dump(Mutation_result,file)
file.close()





