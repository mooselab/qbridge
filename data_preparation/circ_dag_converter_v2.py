import string
import math
import networkx as nx
import rustworkx as rx
import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGInNode, DAGOpNode, DAGOutNode
from qiskit.providers.aer import AerSimulator
from qiskit.providers.fake_provider import FakeMontreal, FakeWashington, FakeSherbrooke, FakeToronto
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements
from torch_geometric.utils.convert import from_networkx
import importlib, json
from functools import lru_cache
from .helper import get_openqasm_gates
import json
import sys
if sys.version_info < (3, 10, 0):
    import importlib_resources as resources
else:
    from importlib import resources  # type: ignore[no-redef]

import logging
logger = logging.getLogger("quantum error mitigation")


GATE_DICT = {item: index for index, item in enumerate(get_openqasm_gates())}
# NUM_ERROR_DATA = 4
NUM_NODE_TYPE = 2 + len(GATE_DICT)

_FAKE_MODS = ["qiskit.providers.fake_provider"]

@lru_cache(None)
def get_backend(device_name: str):
    """Construct a Fake backend instance by name, supporting backends such as FakeToronto."""
    for modname in _FAKE_MODS:
        try:
            mod = importlib.import_module(modname)
            cls = getattr(mod, device_name)
            return cls()
        except (ModuleNotFoundError, AttributeError):
            continue
    raise ValueError(f"Unknown fake backend: {device_name}")


def get_global_features(circ):
    data = torch.zeros((1, 6))
    data[0][0] = circ.depth()
    data[0][1] = circ.width()
    for key in GATE_DICT:
        if key in circ.count_ops():
            data[0][2 + GATE_DICT[key]] = circ.count_ops()[key]

    return data


def to_networkx(dag):
    """Returns a copy of the DAGCircuit in networkx format."""
    G = nx.MultiDiGraph()
    for node in dag._multi_graph.nodes():
        G.add_node(node)
    for node_id in rx.topological_sort(dag._multi_graph):
        for source_id, dest_id, edge in dag._multi_graph.in_edges(node_id):
            G.add_edge(dag._multi_graph[source_id], dag._multi_graph[dest_id], wire=edge)
    return G


def networkx_torch_convert(dag, length):
    myedge = []
    for item in dag.edges:
        myedge.append((item[0], item[1]))
    G = nx.DiGraph()
    G.add_nodes_from(dag._node)
    G.add_edges_from(myedge)
    x = torch.zeros((len(G.nodes()), length))
    try:
        for idx, node in enumerate(G.nodes()):
            x[idx] = dag.nodes[node]["x"]
    except Exception as e:
        print(dag.nodes[node])
    G = from_networkx(G)
    G.x = x
    return G



def get_noise_dict(device_name):
    backend = get_backend(device_name)
    prop = backend.properties().to_dict()

    # ### Added: a small utility to convert values to a target unit.
    def _to_unit(value, unit, target):
        if value is None: return None
        u = (unit or "").lower()
        m = {"s":1e0, "ms":1e-3, "us":1e-6, "µs":1e-6, "ns":1e-9}
        if u not in m or target not in m:
            return float(value)
        # Convert the original value to seconds first, then convert to the target unit
        seconds = float(value) * m[u]
        return seconds / m[target]

    noise_dict = {"qubit": {}, "gate": {}}

    # --- Qubit Level ---
    for i, qubit_prop in enumerate(prop["qubits"]):
        noise_dict["qubit"][i] = {}
        for item in qubit_prop:
            name = item["name"].lower()
            val  = item.get("value")
            unit = item.get("unit", "")
            if name == "t1":
                noise_dict["qubit"][i]["T1"] = _to_unit(val, unit, "us")  # ### CHANGED: unified to μs
            elif name == "t2":
                noise_dict["qubit"][i]["T2"] = _to_unit(val, unit, "us")  # ### CHANGED
            elif name == "readout_error":                               # ### ADDED
                noise_dict["qubit"][i]["readout_error"] = float(val or 0.0)
            elif name == "prob_meas0_prep1":                            # ### ADDED optional
                noise_dict["qubit"][i]["p01"] = float(val or 0.0)
            elif name == "prob_meas1_prep0":                            # ### ADDED optional
                noise_dict["qubit"][i]["p10"] = float(val or 0.0)
            elif name == "readout_length":                              # ### ADDED
                noise_dict["qubit"][i]["readout_length_ns"] = _to_unit(val, unit, "ns")

    # --- Gate Level ---
    for gate_prop in prop["gates"]:
        gate_name = gate_prop["gate"]
        if gate_name not in GATE_DICT:      
            continue
        qubit_list = tuple(sorted(gate_prop["qubits"]))
        if qubit_list not in noise_dict["gate"]:
            noise_dict["gate"][qubit_list] = {}
        info = {}
        # ### ADDED: Extract gate_error / gate_length from `parameters` (convert units to ns)
        for p in gate_prop.get("parameters", []):
            pname = p.get("name", "").lower()
            pval  = p.get("value")
            punit = p.get("unit", "")
            if pname in ("gate_error", "error"):
                info["gate_error"] = float(pval or 0.0)
            elif pname in ("gate_length", "duration"):  # some backends use duration
                info["gate_length_ns"] = _to_unit(pval, punit, "ns")
        # default if not provided
        info.setdefault("gate_error", 0.0)
        info.setdefault("gate_length_ns", 0.0)
        noise_dict["gate"][qubit_list][gate_name] = info

    return noise_dict



def data_generator(node, noise_dict):
    # ### ADDED: Robustly retrieve the qubit index; avoid using `_index` directly
    def _qindex(bit):
        return getattr(bit, "index", getattr(bit, "_index", None))

    try:
        if isinstance(node, DAGInNode):
            qidx = int(_qindex(node.wire))                       # ### CHANGED
            qinfo = dict(noise_dict["qubit"].get(qidx, {}))
            # ### ADDED: Populate required keys for in/out nodes to avoid downstream `KeyError`s
            qinfo.setdefault("readout_error", 0.0)
            qinfo.setdefault("readout_length_ns", 0.0)
            qinfo.setdefault("gate_error", 0.0)
            qinfo.setdefault("gate_len_ns", 0.0)
            return "in", [qidx], [qinfo]

        elif isinstance(node, DAGOutNode):
            qidx = int(_qindex(node.wire))                       # ### CHANGED
            qinfo = dict(noise_dict["qubit"].get(qidx, {}))
            qinfo.setdefault("readout_error", 0.0)
            qinfo.setdefault("readout_length_ns", 0.0)
            qinfo.setdefault("gate_error", 0.0)
            qinfo.setdefault("gate_length_ns", 0.0)
            return "out", [qidx], [qinfo]

        elif isinstance(node, DAGOpNode):
            name = node.name.lower()
            qargs = node.qargs
            qubit_list = [int(_qindex(q)) for q in qargs]        # ### CHANGED
            qubit_list_sorted = tuple(sorted(qubit_list))        # ### ADDED

            # Basic info for each participating qubit (T1/T2/readout)
            per_qubit = []
            for qi in qubit_list:
                base = dict(noise_dict["qubit"].get(qi, {}))
                
                base.setdefault("readout_error", 0.0)
                base.setdefault("readout_length_ns", 0.0)
                per_qubit.append(base)

            # ### ADDED: Gate-level error rate / duration (identical for all qubits involved in the same gate)
            gtable = noise_dict["gate"].get(qubit_list_sorted, {})
            ginfo  = gtable.get(name, {"gate_error": 0.0, "gate_length_ns": 0.0})
            for d in per_qubit:
                d["gate_error"]  = float(ginfo.get("gate_error", 0.0))
                d["gate_length_ns"] = float(ginfo.get("gate_length_ns", 0.0))

            # Special handling for measurement gates: no gate_error/len; keep readout_* (already in per_qubit)
            # No additional branching here—just keep the default values
            return name, qubit_list, per_qubit

        else:
            raise NotImplementedError("Unknown node type")

    except Exception as e:
        logger.warning(e)


def circ_to_dag_with_data(circ, device_name, n_qubit=10):
    # data format:
    # [ node_type onehot (K=NUM_NODE_TYPE) ] +
    # [ qubit_idx multi-hot (Q=n_qubit) ] +
    # [ per_qubit_q1(4) | per_qubit_q2(4) ] +
    # [ flag_is_two_qubit(1) ] +
    # [ time_feats(2) = layer_norm, index_norm ] +
    # [ gate_params(3) = up to 3 angles ]
    # total length = NUM_NODE_TYPE + n_qubit + 14

    circ = circ.copy()
    circ = RemoveBarriers()(circ)
    circ = RemoveFinalMeasurements()(circ)  # Comment out if measurement nodes are needed

    dag_q = circuit_to_dag(circ)
    dag = to_networkx(dag_q)
    dag_list = list(dag.nodes())

    noise_dict = get_noise_dict(device_name)

    def _build_qubit_calib_arrays(noise_dict, n_qubit):
        T1 = np.full(n_qubit, np.inf, dtype=float)   # default inf → idle=exp(-Δt/T)=1
        T2 = np.full(n_qubit, np.inf, dtype=float)
        ro = np.zeros(n_qubit, dtype=float)          # default 0
        for q in range(n_qubit):
            qd = noise_dict["qubit"].get(q, {})
            if "T1" in qd: T1[q] = float(qd["T1"])                 # unity: μs
            if "T2" in qd: T2[q] = float(qd["T2"])                 # unity: μs
            if "readout_error" in qd: ro[q] = float(qd["readout_error"])
        return T1, T2, ro

    # ===== batch-remove useless in→(measure/out) pairs =====
    to_remove = set()
    for node in dag_list:
        if isinstance(node, DAGOpNode) and node.name == 'measure':
            continue
        try:
            node_type, qubit_idxs, noise_info = data_generator(node, noise_dict)
        except Exception:
            continue
        if node_type == "in":
            succnodes = list(dag.succ[node])
            for succnode in succnodes:
                if isinstance(succnode, DAGOpNode) and succnode.name == 'measure':
                    to_remove.add(node); to_remove.add(succnode); continue
                succnode_type, _, _ = data_generator(succnode, noise_dict)
                if succnode_type == "out":
                    to_remove.add(node); to_remove.add(succnode)
    if to_remove:
        dag.remove_nodes_from(to_remove)

    # ===== After deleting nodes, recompute the topological order and layer indices in a single, canonical pass =====
    order = list(nx.topological_sort(dag))
    layer = {}
    for n in order:
        preds = list(dag.predecessors(n))
        layer[n] = 0 if not preds else max(layer[p] for p in preds) + 1
    max_layer = max(layer.values()) if layer else 1

    # ===== Utility: gate duration / readout duration (μs), and 3D angle parameters =====
    def _gate_len_us(info_dict, is_measure=False):
        ns = None
        if isinstance(info_dict, dict):
            ns = info_dict.get("gate_length_ns", None)
            if ns is None and is_measure:
                ns = info_dict.get("readout_length_ns", 0.0)
        return float(ns)/1000.0 if ns is not None else 0.0

    def _gate_params_3(node_obj):
        if isinstance(node_obj, DAGOpNode):
            params = []
            for p in getattr(node_obj.op, "params", []):
                try: params.append(float(p))
                except Exception: params.append(0.0)
            name = node_obj.name.lower()
            if name == "u2":
                params = [math.pi/2] + params[:2]
            params = (params + [0.0]*3)[:3]
            params = [((x + math.pi) % (2*math.pi) - math.pi) for x in params]
            return params
        return [0.0, 0.0, 0.0]

    # ===== ASAP timeline (rolled forward based on gate durations) =====
    timeline = [0.0] * int(n_qubit)   # Current time for each physical qubit (μs)

    D_NODE = NUM_NODE_TYPE + 14

    for topo_idx, node in enumerate(order):
        try:
            node_type, qubit_idxs, noise_info = data_generator(node, noise_dict)
        except Exception:
            continue

        is_meas = (node_type in ("measure", "meas"))

        # --- Node gate duration / error rate (for the timeline and downstream edge features) ---
        if node_type in ("in", "out"):
            L_ns, gate_err = 0.0, 0.0
        else:
            d0 = noise_info[0] if (isinstance(noise_info, list) and noise_info) else {}
            if is_meas:
                L_ns = float(d0.get("readout_length_ns", 0.0))
                gate_err = 0.0
            else:
                L_ns = float(d0.get("gate_length_ns", 0.0))
                gate_err = float(d0.get("gate_error", 0.0))

        # --- ASAP: start_time_us = max(current_time_us over participating qubits) ---
        if qubit_idxs:
            st_us = max(timeline[q] for q in qubit_idxs)
        else:
            st_us = 0.0

        # Update the timeline (in/out nodes do not advance time)
        L_us = L_ns / 1000.0
        if node_type not in ("in", "out"):
            for q in qubit_idxs:
                timeline[q] = st_us + L_us

        # ===== Assemble the node feature vector x =====
        data = torch.zeros(D_NODE, dtype=torch.float32)

        # node_type one-hot
        if node_type == "in":
            data[0] = 1
        elif node_type == "out":
            data[1] = 1
        else:
            data[2 + GATE_DICT[node_type]] = 1

        base = NUM_NODE_TYPE

        # per-qubit 4D ×2： [len/T1, len/T2, gate_error, readout_error]
        def _blk(info_q):
            T1   = float(info_q.get("T1", 0.0)) if isinstance(info_q, dict) else 0.0
            T2   = float(info_q.get("T2", 0.0)) if isinstance(info_q, dict) else 0.0
            rerr = float(info_q.get("readout_error", 0.0)) if isinstance(info_q, dict) else 0.0
            gerr = float(info_q.get("gate_error", 0.0)) if isinstance(info_q, dict) else 0.0
            Lloc = _gate_len_us(info_q, is_measure=is_meas)
            v0 = (Lloc / T1) if T1 > 0 else 0.0
            v1 = (Lloc / T2) if T2 > 0 else 0.0
            return [v0, v1, gerr, rerr]

        if len(qubit_idxs) >= 1:
            data[base : base+4] = torch.tensor(
                _blk(noise_info[0] if isinstance(noise_info, list) and len(noise_info)>=1 else {}),
                dtype=torch.float32
            )
        if len(qubit_idxs) >= 2:
            data[base+4 : base+8] = torch.tensor(
                _blk(noise_info[1] if isinstance(noise_info, list) and len(noise_info)>=2 else {}),
                dtype=torch.float32
            )

        # flag: is_two_qubit
        flg_base = base + 8
        data[flg_base] = 1.0 if len(qubit_idxs) == 2 else 0.0

        # time_feats: Normalized layer index / normalized topological order index
        t_base = flg_base + 1
        layer_norm = (layer.get(node, 0) / max(1, max_layer))
        index_norm = (topo_idx / max(1, len(order)-1))
        data[t_base : t_base+2] = torch.tensor([layer_norm, index_norm], dtype=torch.float32)
        # data[t_base : t_base+2] = torch.tensor([layer.get(node, 0), topo_idx], dtype=torch.float32)

        # gate params (3)
        p_base = t_base + 2
        data[p_base : p_base+3] = torch.tensor(_gate_params_3(node), dtype=torch.float32)

        # Write back node attributes (for edge features / downstream use)
        dag.nodes[node]["x"]              = data
        dag.nodes[node]["gate_length_ns"]    = float(L_ns)
        dag.nodes[node]["gate_error"]     = float(gate_err)
        dag.nodes[node]["layer_idx"]      = int(layer.get(node, 0))
        dag.nodes[node]["start_time_us"]  = float(st_us)

    
    T1_us_arr, T2_us_arr, ro_err_arr = _build_qubit_calib_arrays(noise_dict, n_qubit)

    # Attach it to the graph object so it can be retrieved anytime later
    dag.graph["T1_us"] = T1_us_arr
    dag.graph["T2_us"] = T2_us_arr
    dag.graph["readout_error"] = ro_err_arr
    
    # ---- Relabel: map nodes to 0..N-1 by topological order (replacing `ordering="topological"`) ----
    mapping = {n: i for i, n in enumerate(order)}
    dag = nx.relabel_nodes(dag, mapping, copy=True)

    
    return dag

    return networkx_torch_convert(dag, length=D_NODE)



def get_edge_features_matrix_calib(G, n_qubit,
                                   include_node_redundant=False,
                                   qmh_start=None):
    """
    Return:
      edge_index: LongTensor [2, E]
      edge_attr : FloatTensor [E, d_edge]
    """
    import numpy as np, math, torch

    adj_matrix, node_feature_matrix, nodelist, node_to_idx = graph_to_arrays(G)

    node_gate_len_ns  = np.array([G.nodes[n].get("gate_length_ns", 0.0) for n in nodelist], dtype=float)
    node_gate_error   = np.array([G.nodes[n].get("gate_error", 0.0)     for n in nodelist], dtype=float)
    node_layer_idx    = np.array([G.nodes[n].get("layer_idx", 0)        for n in nodelist], dtype=int)
    node_start_time_us= np.array([G.nodes[n].get("start_time_us", 0.0)  for n in nodelist], dtype=float)

    T1_us         = G.graph["T1_us"]
    T2_us         = G.graph["T2_us"]
    readout_error = G.graph["readout_error"]

    # qubit multi-hot start point
    if qmh_start is None:
        qmh_start = NUM_NODE_TYPE  # Ensure this global/constant exists; otherwise pass it as an argument

    def _safe(a, i, d):
        if a is None or i >= len(a) or a[i] is None:
            return d
        return float(a[i])

    def _qubits_of(node_idx):
        row = node_feature_matrix[node_idx, qmh_start:qmh_start + n_qubit]
        return np.flatnonzero(row > 0.5).tolist()

    # Collect edges in a fixed order to ensure reproducibility
    if hasattr(adj_matrix, "nonzero"):
        srcs, dsts = adj_matrix.nonzero()
    else:
        srcs, dsts = np.where(np.asarray(adj_matrix) == 1)
    edges = sorted([(int(u), int(v)) for u, v in zip(srcs, dsts)])

    # Dimensions: slim edge = 5 (2×idle + Δlayer); full/redundant = 12 (2×6)
    d_edge = 12 if include_node_redundant else 5
    feats_all = []

    max_layer = (max(node_layer_idx) if node_layer_idx is not None else 1) or 1

    for (u, v) in edges:
        inter = sorted(set(_qubits_of(u)).intersection(_qubits_of(v)))  # ≤ 2

        # The duration and error rate upon reaching gate `v`
        len_us = _safe(node_gate_len_ns, v, 0.0) / 1000.0
        gerr   = _safe(node_gate_error,   v, 0.0)

        # idle Δt（μs）
        if node_start_time_us is not None:
            start2 = _safe(node_start_time_us, v, 0.0)
            end1   = _safe(node_start_time_us, u, 0.0) + _safe(node_gate_len_ns, u, 0.0) / 1000.0
            dt = max(0.0, start2 - end1)
        else:
            dt = 0.0

        if include_node_redundant:
            feats = [0.0] * 12
            for i, q in enumerate(inter[:2]):
                T1 = _safe(T1_us, q, float("inf"))
                T2 = _safe(T2_us, q, float("inf"))
                r  = _safe(readout_error, q, 0.0)
                len_over_T1 = 0.0 if not (T1 > 0) else len_us / T1
                len_over_T2 = 0.0 if not (T2 > 0) else len_us / T2
                idle_S_T1 = 1.0 if not (T1 > 0) else math.exp(-dt / T1)
                idle_S_T2 = 1.0 if not (T2 > 0) else math.exp(-dt / T2)
                base = i * 6
                feats[base:base+6] = [len_over_T1, len_over_T2, gerr, idle_S_T1, idle_S_T2, r]
        else:
            # [idle_S_T1_q1, idle_S_T2_q1, idle_S_T1_q2, idle_S_T2_q2, Δlayer_norm]
            feats = [0.0] * 5
            for i, q in enumerate(inter[:2]):
                T1 = _safe(T1_us, q, float("inf"))
                T2 = _safe(T2_us, q, float("inf"))
                idle_S_T1 = 1.0 if not (T1 > 0) else math.exp(-dt / T1)
                idle_S_T2 = 1.0 if not (T2 > 0) else math.exp(-dt / T2)
                base = i * 2
                feats[base:base+2] = [idle_S_T1, idle_S_T2]
            delta_layer = (_safe(node_layer_idx, v, 0.0) - _safe(node_layer_idx, u, 0.0)) / max_layer
            feats[4] = float(delta_layer)

        feats_all.append(feats)

    edge_attr  = torch.tensor(feats_all, dtype=torch.float32)                    # [E,d_edge]
    return edge_attr



def graph_to_arrays(G):
    # Ensure a stable order: use a nodelist sorted by topological order
    nodelist = list(nx.topological_sort(G))
    node_to_idx = {n:i for i,n in enumerate(nodelist)}

    # Adjacency (sparse is preferable; if your downstream requires dense, then call .toarray())
    adj_matrix = np.array(nx.adjacency_matrix(G).todense())

    # Node feature matrix (stack each node's x)
    rows = []
    for n in nodelist:
        x = G.nodes[n]["x"]
        if hasattr(x, "detach"):  # torch.Tensor
            x = x.detach().cpu().numpy()
        rows.append(np.asarray(x, dtype=np.float32))
    node_feature_matrix = np.stack(rows, axis=0)   # [N, D]

    return adj_matrix, node_feature_matrix, nodelist, node_to_idx


