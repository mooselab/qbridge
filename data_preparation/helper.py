from __future__ import annotations

import os
import sys
sys.path.append(os.getcwd() + '/data_preparation')
from typing import Any
import logging
logger = logging.getLogger("qet-predictor")

if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from numpy._typing import NDArray

from pathlib import Path
import os
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def get_path_training_data():
    """Returns the path to the training data folder."""
    return Path(os.getcwd()) / "data"


def get_openqasm_gates():
    """Returns a list of all quantum gates within the openQASM 2.0 standard header."""
    # according to https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/qasm/libs/qelib1.inc
    return [
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "c3x",
        "c3sqrtx",
        "c4x",
        "xx_plus_yy",
        "ecr",
        "reset"
    ]


def dict_to_featurevector(gate_dict: dict[str, int]):
    """Calculates and returns the feature vector of a given quantum circuit gate dictionary."""
    res_dct = dict.fromkeys(get_openqasm_gates(), 0)
    for key, val in dict(gate_dict).items():
        if key in res_dct:
            res_dct[key] = val

    return res_dct


def calc(y_true, y_pred):
    # calculate R-squared
    r_squared = r2_score(y_true, y_pred)
    
    # calculate MSE
    mse = mean_squared_error(y_true, y_pred)
    
    # calculate NMSE
    nmse = mse / np.var(y_true)
    
    print("MSE:", mse)
    print("R-squared:", r_squared)
    print("NMSE:", nmse)


def tvd(p, q):
    return 0.5 * np.abs(p - q).sum()


def HellingerDistance(p, q):
    n = len(p)
    sum_ = 0.0
    for i in range(n):
        sum_ += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum_)
    return result