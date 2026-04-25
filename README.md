## Preparation
conda env create -f environment.yml
conda activate qbridge
pip install -r requirements-ml.txt
export TF_USE_LEGACY_KERAS=True

## QOIN

The code in the `QOIN` folder is copied and revised from the paper *Mitigating Noise in Quantum Software Testing Using Machine Learning* ([link](https://github.com/AsmarMuqeet/QOIN/tree/v1)).

- Run `python QOIN/DataGeneration.py` to generate baseline circuits.  
- Run `python QOIN/MLPTraining.py` to pretrain QOIN’s baseline models.  
- Run `python QOIN/BaselineTuner.py` to generate CUTs and finetune QOIN’s tuning models.  
- Run `python QOIN/EvaluationRQ1.py` to generate testing circuits.  
- Run `python QOIN/CalculatingRQ1.py` to obtain QOIN’s results for RQ1.  
- Run `python QOIN/EvaluationRQ3.py` and `python QOIN/CalculatingRQ3.py` to obtain QOIN’s results for RQ3.  


## QLEAR

The code in the `QLEAR` folder is copied and revised from the paper *A Machine Learning-Based Error Mitigation Approach for Reliable Software Development on IBM’s Quantum Computers* ([link](https://zenodo.org/records/11181417)).

- Run `python QLEAR/DataGeneration_QLEAR_Pretrain.py` to generate baseline circuits.  
- Run `python QLEAR/QLEAR_Pretrain_MLP_ByBackend.py` to pretrain QLEAR’s baseline models.  
- Run `python QLEAR/QLEAR_Finetune_Data_ByBackendFamily.py` to generate CUTs circuits.
- Run `python QLEAR/QLEAR_Finetune_MLP_BySeedBackendFamily.py` to finetune QLEAR’s tuning models.  
- Run `python QLEAR/QLEAR_Test_Data_ByBackendFamily.py` to generate testing circuits.  
- Run `python QLEAR/QLEAR_Evaluate_Hellinger_BySeedBackendFamily.py` to obtain QLEAR’s results for RQ1.  

## QRAFT

The code in the `QRAFT` folder is copied and revised from the paper *Qraft: reverse your quantum circuit and know the correct program output* ([link](https://zenodo.org/records/4527305)).

- Run `python QRAFT/QraftFeatureGeneration.py` to generate baseline circuits.  
- Run `python QRAFT/QraftEDTPretrain.m` to pretrain QRAFT’s baseline models.  
- Run `python QRAFT/QraftFamilyTuneDataGeneration.py` to generate CUTs circuits.
- Run `python QRAFT/QraftEDTFinetune.m` to finetune QRAFT’s tuning models.  
- Run `python QRAFT/QraftTestDataGeneration.py` to generate testing circuits.  
- Run `python QRAFT/QQraftEDTEvaluateTest.m` to obtain QRAFT’s results for RQ1. 


## data_preparation

This folder contains auxiliary scripts for data processing and preprocessing tasks.  
They are used to prepare datasets before running the main experiments and training pipelines.

## transformer

For Q-Bridge **backend-wise model**:  
- Run `pretrain.sh` for pretraining  
- Run `finetune.sh` or `finetune_parallel.sh` for finetuning  
- Run `test.sh` for testing  

For Q-Bridge **general model**:  
- Run `pretrain_full_model.sh` for pretraining  
- Run `finetune_full_model.sh` for finetuning  
- Run `test_full_model.sh` for testing 


## ablation study
For Q-Bridge **backend-wise model**:  
- Run `pretrain_ablation.sh` for pretraining  
- Run `finetune_ablation.sh` for finetuning  
- Run `test_ablation.sh` for testing  

For Q-Bridge **general model**:  
- Run `pretrain_full_model.sh` for pretraining  
- Run `finetune_full_model.sh` for finetuning  
- Run `test_full_model.sh` for testing 

## larger_circuits
- Run `python larger_circuits/BaselineTuner.py` to generate circuits for finetuning and testing.
- Run `python larger_circuits/EvaluationRQ1.py` to generate datasets for finetuning and testing.  
- Refer to `transformer` for finetuning and testing model

### setting
- Remove Edge-Biased Attention: ./transformer/config.yaml -> use_edge_bias: false
- Remove FiLM: ./transformer/config.yaml -> use_film: false
- Remove backend embedding: replace 'import model' with 'import model_v2' for all the files
- Add qubit multi-hot: replace 'import circuit_dag_converter_v2' with 'import circuit_dag_converter' for all the files




