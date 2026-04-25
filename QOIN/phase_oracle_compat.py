import ast
import re
from typing import Dict, List

from qiskit import QuantumCircuit


class _BoolExprEvaluator(ast.NodeVisitor):
    """Safely evaluate a restricted boolean expression."""

    def __init__(self, env: Dict[str, bool]):
        self.env = env

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Name(self, node):
        if node.id not in self.env:
            raise ValueError(f"Unknown variable in oracle expression: {node.id}")
        return bool(self.env[node.id])

    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant in oracle expression: {node.value}")

    def visit_NameConstant(self, node):
        if isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported name constant in oracle expression: {node.value}")

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return not self.visit(node.operand)
        if isinstance(node.op, ast.Invert):
            # Treat "~x" as logical NOT.
            return not self.visit(node.operand)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            return all(self.visit(v) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(self.visit(v) for v in node.values)
        raise ValueError(f"Unsupported boolean operator: {type(node.op).__name__}")

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.BitAnd):
            return bool(left) and bool(right)
        if isinstance(node.op, ast.BitOr):
            return bool(left) or bool(right)
        if isinstance(node.op, ast.BitXor):
            return bool(left) ^ bool(right)

        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    def visit_Compare(self, node):
        raise ValueError("Comparison operators are not supported in oracle expressions.")

    def generic_visit(self, node):
        raise ValueError(f"Unsupported syntax in oracle expression: {type(node).__name__}")


def _translate_expr(expr: str) -> str:
    """
    Keep original oracle syntax and only normalize whitespace.

    Supported syntax:
      ~  logical NOT
      &  logical AND
      |  logical OR
      ^  logical XOR
    """
    return expr.strip()


def _extract_variables(expr: str) -> List[str]:
    """
    Extract variable names and sort them naturally by numeric suffix.
    Examples: x0, x1, v0, v1, a, b, c
    """
    names = sorted(set(re.findall(r"[A-Za-z_]\w*", expr)))

    def sort_key(name: str):
        m = re.match(r"([A-Za-z_]+)(\d+)$", name)
        if m:
            return (m.group(1), int(m.group(2)))
        return (name, -1)

    return sorted(names, key=sort_key)


class PhaseOracle(QuantumCircuit):
    """
    Lightweight replacement for qiskit.circuit.library.PhaseOracle.

    This implementation is exponential in the number of variables, so it is
    intended only for small benchmark circuits.
    """

    def __init__(self, expression: str, synthesizer=None, var_order=None):
        if not isinstance(expression, str):
            raise TypeError("This compatibility PhaseOracle only supports string expressions.")

        self.expression = expression
        self._translated_expr = _translate_expr(expression)
        self.variable_order = list(var_order) if var_order is not None else _extract_variables(expression)

        num_vars = len(self.variable_order)
        super().__init__(num_vars, name="PhaseOracleCompat")

        self._build_phase_oracle()

    def _eval_assignment(self, bit_values: List[int]) -> bool:
        env = {var: bool(bit) for var, bit in zip(self.variable_order, bit_values)}
        tree = ast.parse(self._translated_expr.strip(), mode="eval")
        return bool(_BoolExprEvaluator(env).visit(tree))

    def evaluate_bitstring(self, bitstring: str) -> bool:
        """
        Compatibility helper matching the old PhaseOracle interface.

        We interpret the bitstring in little-endian order to stay consistent
        with the original implementation style used in many Qiskit examples.
        """
        if len(bitstring) != len(self.variable_order):
            raise ValueError(
                f"Bitstring length {len(bitstring)} does not match oracle arity {len(self.variable_order)}."
            )
        bits_le = [int(b) for b in bitstring]
        return self._eval_assignment(bits_le)

    def _build_phase_oracle(self):
        n = len(self.variable_order)

        if n == 0:
            return

        for value in range(2 ** n):
            bits = [(value >> i) & 1 for i in range(n)]  # little-endian
            if not self._eval_assignment(bits):
                continue

            zero_positions = [i for i, b in enumerate(bits) if b == 0]

            for q in zero_positions:
                self.x(q)

            if n == 1:
                self.z(0)
            else:
                target = n - 1
                controls = list(range(n - 1))
                self.h(target)
                self.mcx(controls, target)
                self.h(target)

            for q in zero_positions:
                self.x(q)

    @classmethod
    def from_dimacs_file(cls, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("c")]

        var_count = None
        clauses = []

        for line in lines:
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4 and parts[1].lower() == "cnf":
                    var_count = int(parts[2])
            else:
                lits = [int(x) for x in line.split() if x != "0"]
                if lits:
                    clauses.append(lits)

        if var_count is None:
            raise ValueError("Invalid DIMACS CNF file: missing 'p cnf' header.")

        vars_ = [f"x{i}" for i in range(var_count)]
        clause_exprs = []
        for clause in clauses:
            terms = []
            for lit in clause:
                idx = abs(lit) - 1
                name = vars_[idx]
                terms.append(name if lit > 0 else f"~{name}")
            clause_exprs.append("(" + " | ".join(terms) + ")")

        expr = " & ".join(clause_exprs) if clause_exprs else "True"
        return cls(expr, var_order=vars_)