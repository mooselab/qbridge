import json
import os
from typing import Any, Dict, List

from benchmark_circuits import get_circuit_class_object

SELECTED_FAMILIES = ["ghz", "phase", "addition", "simon", "qft", "similarity"]


def _freeze_input(value: Any):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_input(v) for v in value)
    return value


def _sorted_unique(values: List[Any]) -> List[Any]:
    seen = set()
    unique = []
    for value in values:
        key = _freeze_input(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)

    def sort_key(x: Any):
        if isinstance(x, tuple):
            return (1, repr(x))
        return (0, repr(x))

    return sorted(unique, key=sort_key)


def get_family_input_splits() -> Dict[str, Dict[str, Any]]:
    splits: Dict[str, Dict[str, Any]] = {}
    for family in SELECTED_FAMILIES:
        circuit = get_circuit_class_object(family)
        train_inputs = _sorted_unique(list(circuit.get_inputs()))
        full_inputs = _sorted_unique(list(circuit.get_full_inputs()))
        train_keys = {_freeze_input(v) for v in train_inputs}
        eval_inputs = [v for v in full_inputs if _freeze_input(v) not in train_keys]
        if not eval_inputs:
            raise ValueError(f"No disjoint evaluation inputs available for family '{family}'.")

        splits[family] = {
            "input_type": circuit.key_aurguments["input_type"],
            "train_inputs": train_inputs,
            "eval_inputs": eval_inputs,
            "train_size": len(train_inputs),
            "eval_size": len(eval_inputs),
        }
    return splits


def export_family_split_manifest(output_path: str) -> Dict[str, Dict[str, Any]]:
    splits = get_family_input_splits()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    serializable = {}
    for family, payload in splits.items():
        serializable[family] = {
            "input_type": payload["input_type"],
            "train_inputs": [list(x) if isinstance(x, tuple) else x for x in payload["train_inputs"]],
            "eval_inputs": [list(x) if isinstance(x, tuple) else x for x in payload["eval_inputs"]],
            "train_size": payload["train_size"],
            "eval_size": payload["eval_size"],
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return splits
