import pickle

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp


AVAILABLE_BACKENDS = ("highs", "gurobi")


class GEMQSolver:
    """
    Conduct global ILP bit allocation for GEMQ.

    The problem is a multiple-choice knapsack over all (layer, expert, bit) triples:
    at most ~18k binary variables even for Qwen3-30B-A3B (48 layers x 128 experts x 3
    bit-widths). HiGHS -- which ships with SciPy -- solves that in seconds, so the
    default backend needs no extra install and no license. "gurobi" is kept as an
    optional alternative; note its size-limited free license only fits small models
    such as Mixtral.

    Variable layout: x[(li * num_experts + j) * num_x + ki] is 1 iff expert j of MoE
    layer li (model layer start_layer_idx + li) is assigned x_space[ki] bits.
    """

    def __init__(
        self,
        layer_re_path,
        x_space=(1, 2, 3),
        extra_constr="",
        start_layer_idx=0,
        backend="highs",
    ):
        if backend not in AVAILABLE_BACKENDS:
            raise ValueError(
                f"Unknown ILP backend: {backend!r}. Available: {AVAILABLE_BACKENDS}"
            )

        # load weighted layer reconstruction error as coefficients
        with open(layer_re_path, "rb") as file:
            self.coef = pickle.load(file)

        self.x_space = list(x_space)  # available bit-width candidates
        self.num_moe_layers = len(self.coef) # number of effective MoE layers (exclude dense layers)
        self.num_layers = start_layer_idx + self.num_moe_layers
        self.num_experts = len(self.coef[start_layer_idx]) # assume all layers have the same number of experts
        self.num_x = len(self.x_space)
        print(f"num_layers: {self.num_layers}, num_experts: {self.num_experts}, x_space: {self.x_space}")

        self.extra_constr = extra_constr

        self.start_layer_idx = start_layer_idx

        self.backend = backend
        self.last_objective = None

    @property
    def num_vars(self):
        return self.num_moe_layers * self.num_experts * self.num_x

    def build_objective(self):
        """
        Flatten the per-(layer, expert, bit) errors into the objective vector.

        Returns:
            c: float64 array of length num_vars
        """
        c = np.empty(self.num_vars, dtype=np.float64)
        for li in range(self.num_moe_layers):
            layer_coef = self.coef[self.start_layer_idx + li]
            for j in range(self.num_experts):
                expert_coef = layer_coef[j]
                base = (li * self.num_experts + j) * self.num_x
                for ki, k in enumerate(self.x_space):
                    c[base + ki] = expert_coef[k]
        return c

    def build_constraints(self, total_bits):
        """
        Assemble the constraint matrices in a solver-neutral form.

        All matrices are sparse: the one-hot block alone is (num_moe_layers *
        num_experts) x num_vars, which is 6144 x 18432 for Qwen3-30B-A3B -- 113M
        entries if it were dense.

        Args:
            total_bits: total number of allocated bits (bit budget)
        Returns:
            (A_ub, b_ub): A_ub @ x <= b_ub -- the global bit budget (c0)
            (A_eq, b_eq): A_eq @ x == b_eq -- one bit-width per expert (c1)
            (A_lb, b_lb): A_lb @ x >= b_lb -- the optional c2/c3 constraints
        """
        L, E, K = self.num_moe_layers, self.num_experts, self.num_x
        n = self.num_vars

        # c0: global bit budget. Each (layer, expert) block contributes its K
        # candidate bit-widths, so the row is just x_space tiled L * E times.
        budget_row = np.tile(np.asarray(self.x_space, dtype=np.float64), L * E)
        A_ub = sp.csr_matrix(budget_row.reshape(1, n))
        b_ub = np.array([float(total_bits)])

        # c1: exactly one bit-width per expert. Row r covers the K consecutive
        # variables of block r, which is exactly what this CSR layout encodes.
        A_eq = sp.csr_matrix(
            (np.ones(n), np.arange(n), np.arange(0, n + 1, K)),
            shape=(L * E, n),
        )
        b_eq = np.ones(L * E)

        # c2/c3: at least one expert per layer at each of the top two bit-widths
        if self.extra_constr == "c2c3":
            top_bits = sorted(self.x_space, reverse=True)[:2]
            if len(top_bits) < 2:
                raise ValueError(
                    "extra_constr='c2c3' needs at least two candidate bit-widths, "
                    f"got x_space={self.x_space}"
                )
            rows, cols = [], []
            r = 0
            for k in top_bits:
                ki = self.x_space.index(k)
                for li in range(L):
                    for j in range(E):
                        rows.append(r)
                        cols.append((li * E + j) * K + ki)
                    r += 1
            A_lb = sp.csr_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(r, n)
            )
            b_lb = np.ones(r)
        else:
            A_lb = sp.csr_matrix((0, n))
            b_lb = np.zeros(0)

        return (A_ub, b_ub), (A_eq, b_eq), (A_lb, b_lb)

    def _solve_highs(self, c, ub, eq, lb):
        (A_ub, b_ub), (A_eq, b_eq), (A_lb, b_lb) = ub, eq, lb

        constraints = [
            LinearConstraint(A_ub, -np.inf, b_ub),
            LinearConstraint(A_eq, b_eq, b_eq),
        ]
        if A_lb.shape[0] > 0:
            constraints.append(LinearConstraint(A_lb, b_lb, np.inf))

        res = milp(
            c=c,
            constraints=constraints,
            integrality=np.ones(c.size),
            bounds=Bounds(0, 1),
        )
        if not res.success:
            raise RuntimeError(
                f"HiGHS failed to solve the bit-allocation ILP: {res.message}"
            )
        return res.x

    def _solve_gurobi(self, c, ub, eq, lb):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as e:
            raise ImportError(
                "The 'gurobi' backend requires gurobipy (pip install 'gemq[gurobi]'). "
                "The default 'highs' backend needs no extra install."
            ) from e

        (A_ub, b_ub), (A_eq, b_eq), (A_lb, b_lb) = ub, eq, lb

        try:
            m = gp.Model("ilp")
            x = m.addMVar(shape=c.size, vtype=GRB.BINARY, name="x")
            m.setObjective(c @ x, GRB.MINIMIZE)
            m.addConstr(A_ub @ x <= b_ub, name="c0")
            m.addConstr(A_eq @ x == b_eq, name="c1")
            if A_lb.shape[0] > 0:
                m.addConstr(A_lb @ x >= b_lb, name="c2c3")
            m.optimize()

            if m.Status != GRB.OPTIMAL:
                raise RuntimeError(
                    f"Gurobi finished with status {m.Status} instead of an optimal solution"
                )
            sol = np.asarray(x.X, dtype=np.float64).copy()

            m.dispose()
            gp.disposeDefaultEnv()
        except gp.GurobiError as e:
            raise RuntimeError(
                f"Gurobi failed to solve the bit-allocation ILP: {e}\n"
                "The size-limited free license only fits models with few experts "
                "(e.g. Mixtral). Use the default backend='highs' instead."
            ) from e

        return sol

    def decode(self, x):
        """
        Turn a raw binary solution into {layer_idx: {expert_idx: bits}}.
        """
        sel = np.asarray(x).reshape(self.num_moe_layers, self.num_experts, self.num_x)
        sel = sel.round()  # MILP solvers return values like 0.9999999998

        picked = sel.sum(axis=-1)
        if not np.allclose(picked, 1.0):
            raise RuntimeError(
                "ILP solution does not assign exactly one bit-width to every expert"
            )

        ki = sel.argmax(axis=-1)
        return {
            self.start_layer_idx + li: {
                j: self.x_space[ki[li, j]] for j in range(self.num_experts)
            }
            for li in range(self.num_moe_layers)
        }

    def compute_objective(self, opt_set):
        """
        Recompute the objective value from an allocation.

        Solver-independent on purpose: this is what lets two backends -- or a fresh
        run and a config pickle saved long ago -- be compared without trusting
        either solver's own reported objective.
        """
        return sum(
            self.coef[i][j][bits]
            for i, experts in opt_set.items()
            for j, bits in experts.items()
        )

    def solve_all(self, total_bits):
        """
        Solve an ILP problem for all experts in the model.

        Args:
            total_bits: total number of allocated bits
        Returns:
            opt_set: a dictionary with the following structure:
                {
                    start_layer_idx:  {0: <bit>, 1: <bit>, ..., num_expert-1: <bit>},
                    ...
                    num_layers-1:     {0: <bit>, 1: <bit>, ..., num_expert-1: <bit>},
                }
        """
        c = self.build_objective()
        ub, eq, lb = self.build_constraints(total_bits)

        if self.backend == "highs":
            x = self._solve_highs(c, ub, eq, lb)
        else:
            x = self._solve_gurobi(c, ub, eq, lb)

        opt_set = self.decode(x)

        self.last_objective = self.compute_objective(opt_set)
        used_bits = sum(bits for experts in opt_set.values() for bits in experts.values())
        print(
            f"Obj: {self.last_objective:g} "
            f"(backend: {self.backend}, bits used: {used_bits}/{total_bits:g})"
        )

        return opt_set
