import array
import fractions
import logging
import math
import sys
from typing import Optional, Union, List, Tuple
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister,IBMQ
from qiskit.algorithms import AlgorithmResult, AlgorithmError
from qiskit.circuit import Gate, Instruction, ParameterVector
from qiskit.circuit.library import QFT
from qiskit.providers import Backend
from qiskit.quantum_info import partial_trace
from qiskit.utils import summarize_circuits
from qiskit.utils.arithmetic import is_power
from qiskit.utils.quantum_instance import QuantumInstance
from qiskit.utils.validation import validate_min
from qiskit import Aer, transpile,BasicAer
from Abstract_Interface import *
from qiskit_optimization.applications import Knapsack
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.applications.vertex_cover import VertexCover
import networkx as nx
from qiskit_textbook.tools import simon_oracle
from qiskit.circuit.library import PhaseOracle
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.providers.aer.noise import device




def convert_to_bin(value, total_bits=8):
    if isinstance(value, int):
        binary = bin(value)[2:]
        if len(binary) < total_bits:
            return "0" * (total_bits - len(binary)) + binary
        else:
            return binary
    elif isinstance(value, str):
        binary = ' '.join(format(x, 'b') for x in bytearray(value, 'utf-8'))
        if len(binary) < total_bits:
            return "0" * (total_bits - len(binary)) + binary
        else:
            return binary


def convert_to_int(value):
    if len(value.split()) > 1:
        return -1111
    else:
        return int(value, 2)


def convert_to_str(value):
    return "".join([chr(int(x, 2)) for x in value.split()])

if __name__ == '__main__':
    pass