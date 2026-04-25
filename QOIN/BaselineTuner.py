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

import os

import tensorflow as tf

from benchmark_circuits import *
import random
import pandas as pd
from tqdm import *
import pkgutil
import warnings
import exrex
import math
import time
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.qpy import dump, load


warnings.filterwarnings('ignore')


import numpy as np

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def find_array_params(qc):
    for idx, (inst, qargs, cargs) in enumerate(qc.data):
        arr = [p for p in inst.params
               if isinstance(p, (np.ndarray, list, tuple)) and np.asarray(p).ndim > 0]
        if arr:
            shapes = [np.asarray(p).shape if hasattr(p, "__array__") else type(p) for p in inst.params]
            print(f"[{idx}] {inst.name}  params={inst.params}  shapes={shapes}")



backends = [('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65), ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28), ('FakeCasablanca', 7), ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJakarta', 7), ('FakeJohannesburg', 20), ('FakeKolkata', 27), ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27), ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27), ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27), ('FakeToronto', 27), ('FakeWashington', 127)]
BaselineCircuits,CUTs = train_test_split(get_all_circuits(),train_size=0.4,random_state=13)
SEEDS = [1, 2, 3, 4, 5]

os.makedirs(f"tunning_models", exist_ok=True)

backend_factory = BackendFactory()
backend_executors = {}
backend = backend_factory.initialize_backend()
for bk, qubit_size in backends:
    backend_executors[bk] = backend_factory.initialize_backend(bk)

for cut in CUTs:
    print("Current circuit: ",cut)
    circuit = get_circuit_class_object(cut)
    test_inputs = circuit.get_inputs()

    for bk, qubit_size in backends:
        # os.makedirs(f"../data/baseline_tunning_data/{bk}", exist_ok=True)
        # os.makedirs(f"baseline_tunning_data/{bk}", exist_ok=True)
        # data_rows = []
        # single_row = []
        
        # backend_noise = backend_executors[bk]

        # print("Generating Data For {} Backend".format(bk))
        # print("------------------------------------------")

        # print("Executing CUT circuit ID: {}".format(circuit.key_aurguments["ID"]))
        # # if circuit.key_aurguments["ID"] != 4:
        # #     continue
        # start_time = time.time()

        # for iteration in tqdm(range(100)):
        #     inp = test_inputs[iteration%len(test_inputs)]
        #     try:
        #         ideal = circuit.get_result(backend,inp)
        #         # continue
        #         # print(circ.qasm())
        #         # print(circuit.key_aurguments["circuit"].qasm())
        #         # continue
        #         print(str(circuit.key_aurguments["ID"]) + "_" + str(iteration))
        #         with open(f"../data/baseline_tunning_data/{bk}/{str(circuit.key_aurguments['ID'])}_{str(iteration)}.qasm", "w", encoding="utf-8", newline="\n") as f:
        #             f.write(circuit.key_aurguments["circuit"].qasm())
        #     except MissingOptionalLibraryError as ex:
        #         print(ex)
        #     # continue
        #     ideal = circuit.get_result(backend,inp)
        #     Noise = circuit.get_result(backend_noise,inp)

        #     for outputs in ideal["probability"]:
        #         target_variable_prob = None
        #         target_variable_odds = None
        #         actual_target_prob = outputs["prob"]
        #         all_other_probs = 0
        #         for noise_outputs in Noise["probability"]:
        #             if outputs["bin"] == noise_outputs["bin"]:
        #                 target_variable_prob = noise_outputs["prob"]
        #                 target_variable_odds = noise_outputs["odds"]
        #             else:
        #                 all_other_probs += noise_outputs["prob"]
        #         temp_row = [x for x in single_row]
        #         temp_row.extend([all_other_probs,target_variable_odds,target_variable_prob,actual_target_prob, str(circuit.key_aurguments["ID"]) + "_" + str(iteration)])
        #         data_rows.append(temp_row)
                
        #     #0----0-0-----------0-0-00000000000000-0-0-00000000000000000--0-0----------------
        #     for outputs in Noise["probability"]:
        #         if outputs["bin"] not in [x["bin"] for x in ideal["probability"]]:
        #             target_variable_prob = None
        #             target_variable_odds = None
        #             actual_target_prob = 0
        #             all_other_probs = 0
        #             for noise_outputs in Noise["probability"]:
        #                 if outputs["bin"] == noise_outputs["bin"]:
        #                     target_variable_prob = noise_outputs["prob"]
        #                     target_variable_odds = noise_outputs["odds"]
        #                 else:
        #                     all_other_probs += noise_outputs["prob"]
        #             temp_row = [x for x in single_row]
        #             temp_row.extend([all_other_probs,target_variable_odds,target_variable_prob,actual_target_prob, str(circuit.key_aurguments["ID"]) + "_" + str(iteration)])
        #             data_rows.append(temp_row)


        # # #-=-=-=-=-=-=-==-=-=-=-=-=-=-==-Appending to CSV-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        # columns = []
        # columns.extend(["POF","ODR","POS","Target Value","circuit"])
        # data_frame = pd.DataFrame(data_rows,columns=columns)
        # data_frame.to_csv("baseline_tunning_data/{}/{}_{}.csv".format(bk,bk,cut),index=False)
                
        # Transfer learning only
        for seed in SEEDS:
            print("============================================================")
            print("Current seed: ", seed)
            print("============================================================")
            set_seed(seed)
            os.makedirs(f"tunning_models/seed_{seed}", exist_ok=True)

            bkfile = "baseline_tunning_data/{}/{}_{}.csv".format(bk,bk,cut)
            train_df = pd.read_csv(bkfile)
            train_df = train_df.dropna()
            train_df = train_df[['POF', 'ODR', 'POS', 'Target Value']]

            trn, val, preproc = tabular.tabular_from_df(
                train_df,
                is_regression=True, 
                label_columns='Target Value',
                random_state=seed
            )
            
            print("Loading baseline model for backend ",bk)
            predictor = ktrain.load_predictor('baseline_models/{}_baseline'.format(bk))
            
            model = predictor.model
            
            learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

            learner.lr_find(show_plot=True, max_epochs=16)
            
            print("Strating transfer learning")

            learner.autofit(1e-3,early_stopping=10)

            epochs = len(learner.history.history["loss"])
            trainingloss = learner.evaluate(test_data=trn)[0][-1]
            validationloss = learner.evaluate()[0][-1]

            ktrain.get_predictor(learner.model, preproc).save(
                'tunning_models/seed_{}/{}_{}'.format(seed,bk,cut)
            )






