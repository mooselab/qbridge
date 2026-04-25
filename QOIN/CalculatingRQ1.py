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

import pickle
warnings.filterwarnings('ignore')
from pathlib import Path




def HellingerDistance(p, q):
    n = len(p)
    sum_ = 0.0
    for i in range(n):
        sum_ += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum_)
    return result



RQ1C = defaultdict(lambda:{})
dirpath = Path("evaluation_data/")
pattern = "*.json"
for datafiles in sorted(dirpath.glob(pattern)):

    print(datafiles)
    bk,name = str(datafiles).split("/")[-1].split(".")[0].split("_")
    with open("evaluation_data/{}_{}.json".format(bk,name),"r") as file:
        data = json.load(file)
        
    noise_data = data["noise"]
    ideal_data = data["ideal"]
    

    Filter_HL  = []
    Noise_HL   = []

        
    RQ1C.setdefault(bk, {}).setdefault(name, [])
    # print(RQ1C)
    predictor = ktrain.load_predictor('tunning_models/{}_{}'.format(bk,name))
    for i,(each_input_ideal,each_input_noise) in enumerate(zip(ideal_data,noise_data)):
        # filter_probs = []
        # ideal_probs = []
        for key in each_input_ideal.keys():
            filter_probs = []
            ideal_probs = []
            for ideal_value in each_input_ideal[key]["probability"]:
                ideal_probs.append(ideal_value["prob"])
                found = False
                for values in each_input_noise[key]["probability"]:
                    if ideal_value["bin"]==values["bin"]:
                        all_other_probs = sum([x["prob"] for x in each_input_noise[key]["probability"] if x["bin"]!=values["bin"]])
                        odr = values["odds"]
                        pos = values["prob"]
                        pof = 1-pos
                        temp2 = pd.DataFrame([[pof,odr,pos]],columns=["POF","ODR","POS"])
                        prediction = predictor.predict(temp2)[0]
                        #print(prediction)
                        if prediction[0]<0:
                            filter_probs.append(0)
                        elif prediction[0]>1:
                            filter_probs.append(1)
                        else:
                            filter_probs.append(prediction[0])        
                        found = True
                        break                
            
            PF = np.array(ideal_probs).reshape(-1,1)
            QF = np.array(filter_probs).reshape(-1,1)
            HL_filter = HellingerDistance(PF,QF)[0]
            
            RQ1C[bk][name].append(HL_filter)
            # print(RQ1C)

saveRQ1 = {k: v for k, v in RQ1C.items()}
rqfile = open("./results/saveRQ1.json","wb")
pickle.dump(saveRQ1,rqfile)
rqfile.close()


