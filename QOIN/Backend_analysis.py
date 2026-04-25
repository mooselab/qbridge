import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

import urllib.request
import pandas as pd
import numpy as np
import time
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
from qiskit.exceptions import MissingOptionalLibraryError
warnings.filterwarnings('ignore')
import pickle

from collections import defaultdict
from qiskit import QuantumCircuit, transpile


def safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if len(xs) == 0:
        return np.nan
    return float(np.mean(xs))


def safe_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if len(xs) == 0:
        return np.nan
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def get_backend_num_qubits(backend):
    if hasattr(backend, "num_qubits"):
        return int(backend.num_qubits)
    if hasattr(backend, "configuration"):
        try:
            return int(backend.configuration().num_qubits)
        except Exception:
            pass
    return np.nan


def get_backend_name(backend, fallback="unknown"):
    if hasattr(backend, "name"):
        try:
            return backend.name
        except Exception:
            pass
    if hasattr(backend, "configuration"):
        try:
            return backend.configuration().backend_name
        except Exception:
            pass
    return fallback


def get_coupling_edges(backend):
    edges = []

    # Newer backends may expose coupling_map directly
    if hasattr(backend, "coupling_map") and backend.coupling_map is not None:
        try:
            cm = backend.coupling_map
            if hasattr(cm, "get_edges"):
                edges = list(cm.get_edges())
                return edges
            if isinstance(cm, list):
                return list(cm)
        except Exception:
            pass

    # configuration().coupling_map
    if hasattr(backend, "configuration"):
        try:
            cm = backend.configuration().coupling_map
            if cm is not None:
                return list(cm)
        except Exception:
            pass

    # target.build_coupling_map()
    if hasattr(backend, "target"):
        try:
            cm = backend.target.build_coupling_map()
            if hasattr(cm, "get_edges"):
                return list(cm.get_edges())
        except Exception:
            pass

    return edges


def build_undirected_adj(num_nodes, edges):
    adj = {i: set() for i in range(num_nodes)}
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


def connected_components(adj):
    visited = set()
    comps = []
    for s in adj:
        if s in visited:
            continue
        stack = [s]
        comp = []
        visited.add(s)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in visited:
                    visited.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps


def bfs_distances(adj, start):
    dist = {start: 0}
    queue = [start]
    head = 0
    while head < len(queue):
        x = queue[head]
        head += 1
        for y in adj[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                queue.append(y)
    return dist


def graph_diameter(num_nodes, edges):
    if num_nodes is None or num_nodes == 0:
        return np.nan
    adj = build_undirected_adj(num_nodes, edges)
    comps = connected_components(adj)
    if len(comps) == 0:
        return 0
    # Use largest connected component
    largest = max(comps, key=len)
    if len(largest) <= 1:
        return 0

    diam = 0
    for s in largest:
        dist = bfs_distances(adj, s)
        diam = max(diam, max(dist.values()))
    return diam


def graph_density(num_nodes, edges):
    if num_nodes is None or num_nodes <= 1:
        return np.nan
    undirected = set()
    for u, v in edges:
        if u == v:
            continue
        undirected.add(tuple(sorted((u, v))))
    m = len(undirected)
    return 2.0 * m / (num_nodes * (num_nodes - 1))


def get_backend_properties_obj(backend):
    if hasattr(backend, "properties"):
        try:
            return backend.properties()
        except Exception:
            pass
    return None


def extract_backend_noise_stats(backend):
    props = get_backend_properties_obj(backend)

    readout_errors = []
    t1s = []
    t2s = []
    gate1q_errors = []
    gate2q_errors = []
    gate1q_lengths = []
    gate2q_lengths = []

    if props is None:
        return {
            "avg_readout_error": np.nan,
            "std_readout_error": np.nan,
            "avg_t1": np.nan,
            "avg_t2": np.nan,
            "avg_1q_error": np.nan,
            "std_1q_error": np.nan,
            "avg_2q_error": np.nan,
            "std_2q_error": np.nan,
            "avg_1q_length": np.nan,
            "avg_2q_length": np.nan,
        }

    # qubit-level properties
    try:
        for qubit_props in props.qubits:
            for item in qubit_props:
                if item.name == "readout_error":
                    readout_errors.append(item.value)
                elif item.name == "T1":
                    t1s.append(item.value)
                elif item.name == "T2":
                    t2s.append(item.value)
    except Exception:
        pass

    # gate-level properties
    try:
        for gate in props.gates:
            gate_name = getattr(gate, "gate", None)
            qubits = getattr(gate, "qubits", [])
            n_qubits = len(qubits)

            gerr = None
            glen = None
            for param in getattr(gate, "parameters", []):
                if param.name == "gate_error":
                    gerr = param.value
                elif param.name == "gate_length":
                    glen = param.value

            if n_qubits == 1:
                if gerr is not None:
                    gate1q_errors.append(gerr)
                if glen is not None:
                    gate1q_lengths.append(glen)
            elif n_qubits == 2:
                if gerr is not None:
                    gate2q_errors.append(gerr)
                if glen is not None:
                    gate2q_lengths.append(glen)
    except Exception:
        pass

    return {
        "avg_readout_error": safe_mean(readout_errors),
        "std_readout_error": safe_std(readout_errors),
        "avg_t1": safe_mean(t1s),
        "avg_t2": safe_mean(t2s),
        "avg_1q_error": safe_mean(gate1q_errors),
        "std_1q_error": safe_std(gate1q_errors),
        "avg_2q_error": safe_mean(gate2q_errors),
        "std_2q_error": safe_std(gate2q_errors),
        "avg_1q_length": safe_mean(gate1q_lengths),
        "avg_2q_length": safe_mean(gate2q_lengths),
    }


def get_backend_static_stats(backend, backend_name):
    num_qubits = get_backend_num_qubits(backend)
    edges = get_coupling_edges(backend)
    density = graph_density(num_qubits, edges)
    diameter = graph_diameter(num_qubits, edges)

    out = {
        "backend": backend_name,
        "num_qubits": num_qubits,
        "num_coupling_edges": len(edges),
        "topology_density": density,
        "topology_diameter": diameter,
    }
    out.update(extract_backend_noise_stats(backend))
    return out


def list_qasm_files(root_dir):
    qasm_files = []
    if not os.path.isdir(root_dir):
        return qasm_files
    for fn in os.listdir(root_dir):
        if fn.endswith(".qasm"):
            qasm_files.append(os.path.join(root_dir, fn))
    qasm_files.sort()
    return qasm_files


def get_instruction_duration(inst):
    # Qiskit instruction may directly expose duration
    try:
        if hasattr(inst.operation, "duration") and inst.operation.duration is not None:
            return float(inst.operation.duration)
    except Exception:
        pass
    try:
        if hasattr(inst, "duration") and inst.duration is not None:
            return float(inst.duration)
    except Exception:
        pass
    return None


def estimate_idle_stats(transpiled_qc):
    """
    A simple proxy:
    idle_total = total_duration * num_qubits - sum(active_time_per_qubit)
    If duration is unavailable, return NaN.
    """
    try:
        total_duration = transpiled_qc.duration
    except Exception:
        total_duration = None

    if total_duration is None:
        return {
            "total_duration": np.nan,
            "idle_total": np.nan,
            "idle_avg_per_qubit": np.nan,
        }

    num_qubits = transpiled_qc.num_qubits
    active = np.zeros(num_qubits, dtype=float)

    for inst in transpiled_qc.data:
        dur = get_instruction_duration(inst)
        if dur is None:
            continue
        try:
            qidxs = [transpiled_qc.find_bit(q).index for q in inst.qubits]
        except Exception:
            qidxs = []
        for q in qidxs:
            active[q] += dur

    idle_total = float(total_duration) * num_qubits - float(np.sum(active))
    idle_avg = idle_total / num_qubits if num_qubits > 0 else np.nan

    return {
        "total_duration": float(total_duration),
        "idle_total": idle_total,
        "idle_avg_per_qubit": idle_avg,
    }


def get_transpiled_circuit_stats(qc, backend, backend_name, qasm_path):
    try:
        tqc = transpile(qc, backend=backend, optimization_level=0)
    except Exception as ex:
        return {
            "backend": backend_name,
            "qasm_file": os.path.basename(qasm_path),
            "transpile_success": False,
            "error_msg": str(ex),
        }

    op_counts = tqc.count_ops()
    one_q_count = 0
    two_q_count = 0
    measure_count = 0
    barrier_count = 0

    for opname, cnt in op_counts.items():
        if opname == "measure":
            measure_count += int(cnt)
        elif opname == "barrier":
            barrier_count += int(cnt)
        else:
            # Count by qubit arity from instructions
            pass

    # Count 1q / 2q ops by iterating instructions
    for inst in tqc.data:
        try:
            nq = len(inst.qubits)
        except Exception:
            nq = 0
        opname = inst.operation.name
        if opname == "measure":
            continue
        if opname == "barrier":
            continue
        if nq == 1:
            one_q_count += 1
        elif nq == 2:
            two_q_count += 1

    idle_stats = estimate_idle_stats(tqc)

    row = {
        "backend": backend_name,
        "qasm_file": os.path.basename(qasm_path),
        "transpile_success": True,
        "num_qubits": tqc.num_qubits,
        "depth": tqc.depth(),
        "size": tqc.size(),
        "width": tqc.width(),
        "num_clbits": tqc.num_clbits,
        "one_qubit_ops": one_q_count,
        "two_qubit_ops": two_q_count,
        "swap_count": int(op_counts.get("swap", 0)),
        "cx_count": int(op_counts.get("cx", 0)),
        "ecr_count": int(op_counts.get("ecr", 0)),
        "rz_count": int(op_counts.get("rz", 0)),
        "sx_count": int(op_counts.get("sx", 0)),
        "x_count": int(op_counts.get("x", 0)),
        "measure_count": measure_count,
        "barrier_count": barrier_count,
    }
    row.update(idle_stats)
    return row


def summarize_circuit_stats(df):
    if len(df) == 0:
        return {}
    out = {
        "num_circuits": len(df),
        "transpile_success_rate": df["transpile_success"].mean() if "transpile_success" in df.columns else np.nan,
    }

    numeric_cols = [
        "depth", "size", "width", "num_clbits",
        "one_qubit_ops", "two_qubit_ops", "swap_count",
        "cx_count", "ecr_count", "rz_count", "sx_count", "x_count",
        "measure_count", "barrier_count",
        "total_duration", "idle_total", "idle_avg_per_qubit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            good = df[col].dropna()
            out[f"avg_{col}"] = good.mean() if len(good) else np.nan
            out[f"std_{col}"] = good.std(ddof=1) if len(good) > 1 else 0.0 if len(good) == 1 else np.nan
    return out


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



backends = [('FakeAlmaden', 20), ('FakeBoeblingen', 20), ('FakeBrooklyn', 65), ('FakeCairo', 27), ('FakeCambridge', 28), ('FakeCambridgeAlternativeBasis', 28), ('FakeCasablanca', 7), ('FakeGuadalupe', 16), ('FakeHanoi', 27), ('FakeJakarta', 7), ('FakeJohannesburg', 20), ('FakeKolkata', 27), ('FakeLagos', 7), ('FakeManhattan', 65), ('FakeMontreal', 27), ('FakeMumbai', 27), ('FakeNairobi', 7), ('FakeParis', 27), ('FakeRochester', 53), ('FakeSingapore', 20), ('FakeSydney', 27), ('FakeToronto', 27), ('FakeWashington', 127)]


backend_factory = BackendFactory()
backend = backend_factory.initialize_backend()
backend_executor = {}
for bk, qubit_size in tqdm(backends):
    backend_executor[bk] = backend_factory.initialize_backend(bk)


backend_static_rows = []
backend_circuit_rows = []
backend_summary_rows = []

# Collect static backend stats first
for bk, qubit_size in backends:
    backend_obj = backend_executor[bk]
    static_stats = get_backend_static_stats(backend_obj, bk)
    backend_static_rows.append(static_stats)

backend_static_df = pd.DataFrame(backend_static_rows)
os.makedirs("../analysis", exist_ok=True)
backend_static_df.to_csv("../analysis/backend_static_summary.csv", index=False)

# Then compile circuits and collect transpilation stats
for bk, qubit_size in tqdm(backends):
    circuit_dataset = f"../data/evaluation_data/{bk}"
    print("Generating Data For {} Backend".format(bk))
    print("------------------------------------------")

    backend_obj = backend_executor[bk]
    qasm_files = list_qasm_files(circuit_dataset)

    if len(qasm_files) == 0:
        print(f"[WARN] No qasm files found in {circuit_dataset}")
        continue

    per_backend_rows = []

    for qasm_path in tqdm(qasm_files, desc=f"{bk} qasm"):
        try:
            qc = QuantumCircuit.from_qasm_file(qasm_path)
        except Exception as ex:
            row = {
                "backend": bk,
                "qasm_file": os.path.basename(qasm_path),
                "transpile_success": False,
                "error_msg": f"qasm_read_error: {str(ex)}",
            }
            backend_circuit_rows.append(row)
            per_backend_rows.append(row)
            continue

        row = get_transpiled_circuit_stats(qc, backend_obj, bk, qasm_path)
        backend_circuit_rows.append(row)
        per_backend_rows.append(row)

    per_backend_df = pd.DataFrame(per_backend_rows)
    if len(per_backend_df) > 0:
        summary = {"backend": bk}
        summary.update(summarize_circuit_stats(per_backend_df))
        backend_summary_rows.append(summary)

backend_circuit_df = pd.DataFrame(backend_circuit_rows)
backend_summary_df = pd.DataFrame(backend_summary_rows)

backend_circuit_df.to_csv("../analysis/backend_circuit_stats.csv", index=False)
backend_summary_df.to_csv("../analysis/backend_transpile_summary.csv", index=False)

# Merge static + transpile-level summary for convenience
merged_backend_df = backend_static_df.merge(backend_summary_df, on="backend", how="left")
merged_backend_df.to_csv("../analysis/backend_failure_analysis_features.csv", index=False)

print("\nSaved files:")
print("  ../analysis/backend_static_summary.csv")
print("  ../analysis/backend_circuit_stats.csv")
print("  ../analysis/backend_transpile_summary.csv")
print("  ../analysis/backend_failure_analysis_features.csv")





    

            
            
