import os
import pdb
import pickle
import random
import sys
from typing import Callable, List, Optional, Tuple, Any, TYPE_CHECKING

import logging
logger = logging.getLogger("qet-predictor")

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import scipy.signal
import torch
from torchpack.datasets.dataset import Dataset

__all__ = ["CircDataset", "Circ"]

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

if TYPE_CHECKING:
    from numpy._typing import NDArray


random.seed(1234)


def get_path_training_data():
    """Returns the path to the training data folder."""
    return Path(os.getcwd()) / "data"


class CircDataset:
    def __init__(self, split_ratio: List[float], mode, backend, cut, extra_data, shuffle=True):
        super().__init__()
        self.split_ratio = split_ratio
        self.raw = {}
        self.mean = {}
        self.std = {}

        if mode == "pretrain":
            self.mode = "baseline_training_data"
        if mode == "finetune":
            self.mode = "baseline_tunning_data"
        if mode == "test":
            self.mode = "evaluation_data"
        
        self.backend = backend
        self.cut = cut
        self.extra_data = extra_data

        self.shuffle = shuffle

        self._load()
        self._preprocess()
        self._split()

        self.instance_num = len(self.raw["dataset"])
        

    def _load(self):
        self.raw["dataset"] = self.load_training_data()
        if self.shuffle:
            random.shuffle(self.raw["dataset"])


    def _preprocess(self):
        pass
        

    def _split(self):
        instance_num = len(self.raw["dataset"])
        split_train = self.split_ratio[0]
        split_valid = self.split_ratio[0] + self.split_ratio[1]

        self.raw["train"] = self.raw["dataset"][: int(split_train * instance_num)]
        self.raw["valid"] = self.raw["dataset"][
            int(split_train * instance_num) : int(split_valid * instance_num)
        ]
        self.raw["test"] = self.raw["dataset"][int(split_valid * instance_num) :]
        # self.raw["test"] = self.raw["dataset"][int(split_train * instance_num) :]

    def get_data(self, device, split):
        return [data.to(device) for data in self.raw[split]]

    def __getitem__(self, index: int):
        data_this = {"dag": self.raw["dataset"][index]}
        return data_this

    def __len__(self) -> int:
        return self.instance_num
    
    def load_training_data(self):
        """Loads and returns the training data from the training data folder.
        """
        # RQ 1
        # file_path = f"../data/{self.mode}/{self.backend}/G_list"
        file_path = f"../larger_circuits/data/{self.mode}/{self.backend}/G_list"
        # RQ 3
        # file_path = f"../data/testing_data/{self.backend}/G_list"
        if self.cut:
            file_path += "_" + self.cut + ".pkl"
        else:
            file_path += ".pkl"
        file = open(file_path, "rb")
        print("file_path:", file_path)
        
        training_data = pickle.load(file)
        file.close()
        return training_data


class Circ(Dataset):
    def __init__(
        self,
        root: str,
        split_ratio: List[float]
    ):
        self.root = root

        super().__init__(
            {
                split: CircDataset(
                    root=root,
                    split=split,
                    split_ratio=split_ratio,
                )
                for split in ["train", "valid", "test"]
                # for split in ["train", "test"]
            }
        )

