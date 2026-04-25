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

files = [x for x in os.listdir("baseline_training_data/") if "_" not in x]
for datafile in tqdm(files):
    bkfile = 'baseline_training_data/{}'.format(datafile)
    start_time = time.time()
    train_df = pd.read_csv(bkfile)
    train_df = train_df.dropna()
    train_df = train_df[['POF', 'ODR', 'POS', 'Target Value']]
    # print(train_df)
    train_df,test_df = train_test_split(train_df,train_size=0.8,test_size=0.2,random_state=42)
    trn, val, preproc = tabular.tabular_from_df(train_df, is_regression=True, 
                                                 label_columns='Target Value', random_state=42)
    # print(trn, val, preproc)
    # continue

    model = tabular.tabular_regression_model('mlp', trn)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=16)

    learner.lr_find(show_plot=True, max_epochs=16)

    learner.autofit(1e-3)

    epochs = len(learner.history.history["loss"])


    trainingloss = learner.evaluate(test_data=trn)[0][-1]

    validationloss = learner.evaluate()[0][-1]


    ktrain.get_predictor(learner.model, preproc).save('baseline_models/{}_baseline'.format(bkfile.split("/")[-1].split(".")[0]))
    # reload Predictor and extract model
    predictor = ktrain.load_predictor('baseline_models/{}_baseline'.format(bkfile.split("/")[-1].split(".")[0]))

