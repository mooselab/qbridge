
from copy import deepcopy

import numpy as np
import torch
import math
from torch_geometric.loader import DataLoader
from torchpack.utils.config import configs

import builder_v2 as builder
import os

from scipy.stats import chi2 as chi2_dist

def uof_no_bin_len_only(pred_list, ideal_list) -> str:
    return "P" if len(pred_list) == len(ideal_list) else "F"

def wodf_from_prob_arrays_no_norm(pred_list, ideal_list, p_thresh: float = 0.01) -> str:
    if len(pred_list) != len(ideal_list) or len(pred_list) == 0:
        return "F"
    if len(pred_list) == 1 and pred_list[0] - ideal_list[0] <= 0.1:
        return "P"

    p = np.asarray(pred_list, dtype=float)
    q = np.asarray(ideal_list, dtype=float)

    q = np.where(q == 0.0, 1e-12, q)
    chi2_stat = np.sum((p - q) ** 2 / q)
    df = len(p) - 1
    pval = float(chi2_dist.sf(chi2_stat, df))
    return "P" if pval >= p_thresh else "F"


def HellingerDistance(p, q):
    n = len(p)
    sum_ = 0.0
    for i in range(n):
        sum_ += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum_)
    return result


class _GraphSample:
    """x/edge_index/edge_attr"""
    def __init__(self, x, edge_index, edge_attr):
        self.x = x; self.edge_index = edge_index; self.edge_attr = edge_attr

def _make_x_obs_from_row(row, shots=1024, eps=1e-6):
    """row = {'POS','POF','Target Value', ...}
       return: x_obs[4] = [POS, POF, logit(POS), 1/sqrt(shots)], y
    """
    pos = float(row["POS"])
    pos = max(min(pos, 1 - eps), eps)
    pof = float(row.get("POF", 1.0 - pos))
    logit = math.log(pos / (1.0 - pos))
    inv_sqrt_shots = 1.0 / math.sqrt(float(shots))
    x_obs = torch.tensor([pos, pof, logit, inv_sqrt_shots], dtype=torch.float32)
    y = torch.tensor(float(row["Target Value"]), dtype=torch.float32)
    return x_obs, y



class trainer:
    def __init__(self, model, device, criterion, optimizer, scheduler, loaders,
                backend, mode, cut=None, extra_data=None, table_by_cirid=None, xobs_std=None ,default_shots=1024):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.best = 1e10
        self.best_params = None
        self.training_data = {}
        self.backend = backend
        self.mode = mode
        self.cut = cut
        self.extra_data = extra_data

        assert table_by_cirid is not None, "require table_by_cirid: cir_id -> rows"
        self.table_by_cirid = table_by_cirid
        self.xobs_std = xobs_std 
        self.default_shots = default_shots

    
    def _forward_with_cached_h(self, batch, train_encoder=True):
        """
        Returns:
        - pred: [B_total]
        - y: [B_total]
        - cir_ids_exp (List[str]): expanded cir_id list, one entry per state
        """
        # PyG Batch -> list[Data]
        data_list = batch.to_data_list()

        # 1) Run the encoder only once per circuit
        graphs_unique, cir_ids = [], []
        for d in data_list:
            graphs_unique.append(_GraphSample(
                x=d.x.to(self.device),
                edge_index=d.edge_index.to(self.device),
                edge_attr=d.edge_attr.to(self.device)
            ))
            cir_ids.append(d.cir_id)

        # Whether the encoder participates in backpropagation:
        #   Pretraining → True
        #   Fine-tuning head → False
        if train_encoder:
            h_list = self.model.encoder(graphs_unique)          # [B_unique, H]
        else:
            was_training = getattr(self.model.encoder, "training", False) if hasattr(self.model, "encoder") else False
            if hasattr(self.model, "encoder"):
                self.model.encoder.eval()
            with torch.no_grad():
                h_list = self.model.encoder(graphs_unique)
            # Restore the state from before entering the function
            if hasattr(self.model, "encoder"):
                self.model.encoder.train(was_training)

        # 2) Expand multiple states: reuse the same h_graph, generate x_obs/y for each state,
        #    and concatenate the backend vector
        h_all, x_obs_all, y_all, cir_ids_exp = [], [], [], []
        for h, cir_id, d in zip(h_list, cir_ids, data_list):
            rows = self.table_by_cirid[cir_id]   # The multiple states corresponding to this circuit
            shots = getattr(d, "shots", self.default_shots)
            for row in rows:
                xo, y = _make_x_obs_from_row(row, shots=shots)
                h_all.append(h)
                x_obs_all.append(xo.to(self.device))
                y_all.append(y.to(self.device))
                cir_ids_exp.append(cir_id)

        h_all     = torch.stack(h_all, dim=0)        # [B_total, H]
        x_obs_all = torch.stack(x_obs_all, dim=0)    # [B_total, 4]
        if self.xobs_std is not None:
            x_obs_all = self.xobs_std(x_obs_all)
        y_all     = torch.stack(y_all, dim=0)        # [B_total]

        # 3) Run only the head (FiLM head)
        pred = self.model.head(h_all, x_obs_all).view(-1)  # [B_total]
        return pred, y_all, cir_ids_exp


    def train(self):
        self.model.train()
        self.training_data["train_loss"] = []
        self.training_data["val_error"] = []

        # Whether to train the encoder: check if any encoder parameter has requires_grad=True
        train_encoder = any(p.requires_grad for p in getattr(self.model, "encoder", self.model).parameters())

        for epoch in range(configs.num_epochs):
            # Key point: during fine-tuning, ensure the encoder stays in eval mode 
            # # (disable Dropout/BN statistics updates)
            if not train_encoder and hasattr(self.model, "encoder"):
                self.model.encoder.eval()

            loss_sum = 0.0
            state_n = 0
            for batch in self.loaders["train"]:
                self.optimizer.zero_grad()
                pred, y, _ = self._forward_with_cached_h(batch, train_encoder=train_encoder)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item() * y.numel()
                state_n += y.numel()
            self.scheduler.step()
            train_rmse = (loss_sum / max(1, state_n)) ** 0.5
            print(f"[{epoch + 1} / {configs.num_epochs}],sqrtloss={train_rmse} \r", end="")
            self.training_data["train_loss"].append(train_rmse)
            if epoch % 5 == 0:
                val_error = self.valid()
                self.save_best(val_error)
                self.training_data["val_error"].append(val_error)
        # model_path = f"../model/{self.mode}/{self.backend}/"
        model_path = f"../model/{self.mode}/full_model/"
        if self.cut:
            model_path += self.cut + "/"
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.best_params, model_path + "model.pth")
        print("\n")


    def save_best(self, loss):
        if loss < self.best:
            self.best = loss
            self.best_params = deepcopy(self.model.state_dict())

    def valid(self):
        self.model.eval()
        se, state_n = 0.0, 0
        with torch.no_grad():
            for batch in self.loaders["valid"]:
                pred, y, _ = self._forward_with_cached_h(batch, train_encoder=False)
                se += ((pred - y) ** 2).sum().item()
                state_n += y.numel()
        rmse = (se / max(1, state_n)) ** 0.5
        print(f"\t\t\t\t\t\t val_error:{rmse} \r", end="")
        return rmse
    

    def saveall(self):
        mydict = {}
        mydict["test_pred"] = self.training_data["test_pred"]
        mydict["test_y"] = self.training_data["test_y"]
        mydict["test_cir_id"] = self.training_data["test_cir_id"]
        mydict["test_hld_per_circuit"] = self.training_data["test_hld_per_circuit"]
        mydict["test_uow_flag_per_circuit"] = self.training_data["test_uow_flag_per_circuit"]

        # RQ1
        # all_path = f"../model/evaluation_data/{self.backend}/"
        # RQ3
        # all_path = f"../model/testing_data/{self.backend}/"

        # RQ2
        all_path = f"../model/evaluation_data/full_model/"
        # RQ3
        # all_path = f"../model/testing_data/full_model/"

        if self.cut:
            all_path += self.cut + "/"
        os.makedirs(all_path, exist_ok=True)
        all_path += "all.pth"
        torch.save(mydict, all_path)


    def test(self):
        self.training_data["test_pred"] = np.array([])
        self.training_data["test_y"] = np.array([])
        self.training_data["test_cir_id"] = np.array([])
        self.training_data["test_hld_per_circuit"] = {}
        self.training_data["test_uow_flag_per_circuit"] = {} 
        self.test_error = 0.0

        print(len(self.loaders["test"].dataset))
        if len(self.loaders["test"].dataset) <= 1:
            return
        self.model.eval()

        se, state_n = 0.0, 0
        bucket = {}  # cid -> {"pred": [], "ideal": []}

        with torch.no_grad():
            for batch in self.loaders["test"]:
                pred, y, cir_ids_exp = self._forward_with_cached_h(batch, train_encoder=False)

                self.training_data["test_pred"] = np.concatenate(
                    (self.training_data["test_pred"], pred.cpu().numpy())
                )
                self.training_data["test_y"] = np.concatenate(
                    (self.training_data["test_y"], y.cpu().numpy())
                )
                # Expanded cir_id (aligned by state)
                self.training_data["test_cir_id"] = np.concatenate(
                    (self.training_data["test_cir_id"], np.array(cir_ids_exp))
                )
                se += ((pred - y) ** 2).sum().item()
                state_n += y.numel()

                # Aggregate to circuit level
                for cid, p_i, y_i in zip(cir_ids_exp, pred, y):
                    if p_i < 0:
                        p_i = 0
                    if p_i > 1:
                        p_i = 1
                    d = bucket.setdefault(str(cid), {"pred": [], "ideal": []})
                    d["pred"].append(float(p_i))
                    d["ideal"].append(float(y_i))
        
        self.test_error = (se / max(1, state_n)) ** 0.5

        # Circuit-level metrics
        hld_vals = []
        for cid, d in bucket.items():
            p_vec = np.asarray(d["pred"], dtype=float)
            q_vec = np.asarray(d["ideal"], dtype=float)
            print('p_vec', p_vec)
            print('q_vec', q_vec)

            # Hellinger distance
            hld = HellingerDistance(p_vec, q_vec)

            self.training_data["test_hld_per_circuit"][cid] = hld

            hld_vals.append(hld)

            uof = uof_no_bin_len_only(d["pred"], d["ideal"])
            wodf = wodf_from_prob_arrays_no_norm(d["pred"], d["ideal"], p_thresh=0.01)
            flag = 1 if (uof == "F" or wodf == "F") else 0

            self.training_data["test_uow_flag_per_circuit"][cid] = {
                "Uof": uof, "Wodf": wodf, "flag": flag
            }
        
        self.training_data["test_hld_mean"] = float(np.mean(hld_vals)) if hld_vals else 0.0
