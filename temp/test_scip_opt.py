import numpy as np
from itertools import combinations
from functools import partial

# from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize
from scipy.linalg import null_space

# import cobyqa
from qiskit.quantum_info import random_statevector, Statevector

# https://stackoverflow.com/questions/55132107/scipy-fitting-with-parameters-in-a-vector
from operator import add

from pyscipopt import Model, quicksum
from pyscipopt.recipes.nonlinear import set_nonlinear_objective


# TODO Come up with a better class name
class NullSpaceSearchProblem:
    def __init__(self, num_qubits, num_states):
        assert num_qubits > 0
        assert num_states > 1
        self.num_qubits = num_qubits
        self.num_states = num_states
        self.num_amps = 2**num_qubits
        self.num_ops = num_states
        self.num_basis = 2**num_qubits - (num_states - 1)
        self.num_vars_per_op = self.num_basis * 2
        self.num_vars = self.num_ops * self.num_basis * 2
        self.num_coeffs = self.num_vars * self.num_amps
        self.states = []
        self.null_spaces = []
        self.null_coeffs = []
        self.constraints = []
        self.x0 = 0
        self.x = {}
        self.model = Model("NSSP")

    @staticmethod
    def gen_states(
        num_qubits: int,
        num_states: int,
        **kwargs,
    ):
        assert num_qubits > 0
        assert num_states > 1
        states = []
        # TODO Consider setting seeds from the arguments
        states = [random_statevector(2**num_qubits, seed=_) for _ in range(num_states)]
        return states

    def set_states(self, states=None):
        if self.states != []:
            print(
                "Warning: The states are already set. This method call will do nothing."
            )
            return
        elif states is None:
            print("Info: No states are provided. The states will be generated.")
            self.states = self.gen_states(
                num_qubits=self.num_qubits,
                num_states=self.num_states,
            )
        else:
            self.states = states.copy()
        return

    def find_null_spaces(self):
        null_spaces = []
        # Exclude the target state
        for j in range(self.num_states):
            A = []
            for i in range(self.num_states):
                if i != j:
                    A.append(self.states[i].data)

            null_s = null_space(np.array(A))
            null_spaces.append(null_s)

            for i in range(self.num_basis):
                basis = null_s[:, i]
                basis_real = [np.real(num) for num in basis]
                basis_imag = [np.imag(num) for num in basis]
                self.null_coeffs.append(basis_real)
                self.null_coeffs.append(basis_imag)
        # Recursively define the constraints
        # a = np.dot(states[2].data, null_s[:, 0])
        # b = np.dot(states[2].data, null_s[:, 1])
        # Check null basis vdot states
        self.null_spaces = null_spaces.copy()

    def whole_vec_real(self, op_idx, x):
        # Assume null_coeffs is not empty
        # Linear combination of the basis vectors
        # to form a different unit vector in the null space
        sum_real = [0 for _ in range(self.num_amps)]
        for i in range(self.num_basis):
            idx = op_idx * self.num_vars_per_op + 2 * i
            v_real = x[idx]
            v_imag = x[idx + 1]
            c_real = self.null_coeffs[idx]
            c_imag = self.null_coeffs[idx + 1]
            # print(i)
            sum_real = map(
                add,
                sum_real,
                [c_real[_] * v_real - c_imag[_] * v_imag for _ in range(self.num_amps)],
            )
        return list(sum_real)

    def whole_vec_imag(self, op_idx, x):
        # Assume null_coeffs is not empty
        # Linear combination of the basis vectors
        # to form a different unit vector in the null space
        sum_imag = [0 for _ in range(self.num_amps)]
        for i in range(self.num_basis):
            idx = op_idx * self.num_vars_per_op + 2 * i
            v_real = x[idx]
            v_imag = x[idx + 1]
            c_real = self.null_coeffs[idx]
            c_imag = self.null_coeffs[idx + 1]
            # print(i)
            sum_imag = map(
                add,
                sum_imag,
                [c_real[_] * v_imag + c_imag[_] * v_real for _ in range(self.num_amps)],
            )
        return list(sum_imag)

    def mult_vec_and_vec_real(self, op_idx0, op_idx1, x):
        vec0_real = np.array(self.whole_vec_real(op_idx0, x))
        vec0_imag = np.array(self.whole_vec_imag(op_idx0, x))
        vec1_real = np.array(self.whole_vec_real(op_idx1, x))
        vec1_imag = np.array(self.whole_vec_imag(op_idx1, x))
        return np.inner(vec0_real, vec1_real) + np.inner(vec0_imag, vec1_imag)

    def mult_vec_and_vec_imag(self, op_idx0, op_idx1, x):
        vec0_real = np.array(self.whole_vec_real(op_idx0, x))
        vec0_imag = np.array(self.whole_vec_imag(op_idx0, x))
        vec1_real = np.array(self.whole_vec_real(op_idx1, x))
        vec1_imag = np.array(self.whole_vec_imag(op_idx1, x))
        return np.inner(vec0_real, vec1_imag) - np.inner(vec0_imag, vec1_real)

    def op_con(self, op_idx, x):
        # TODO Preserve LC of unit = unit vector
        s = 0
        for j in range(self.num_basis):
            idx = op_idx * self.num_vars_per_op + 2 * j
            s += x[idx] ** 2 + x[idx + 1] ** 2
        return s

    def con(self, i, x):
        return x[2 * i] ** 2 + x[2 * i + 1] ** 2

    def build_cons(self):
        for i in range(self.num_vars):
            self.x[i] = self.model.addVar(vtype="C", name="x(%s)" % i)
        self.model.data = self.x
        for i, j in combinations(list(range(self.num_ops)), 2):
            self.model.addCons(
                self.mult_vec_and_vec_real(i, j, self.x) == 0,
                "mult_v_v_real(%s,%s)" % (i, j),
            )
            self.model.addCons(
                self.mult_vec_and_vec_imag(i, j, self.x) == 0,
                "mult_v_v_imag(%s,%s)" % (i, j),
            )

        for i in range(self.num_ops):
            self.model.addCons(self.op_con(i, self.x) == 1, "op_con(%s)" % (i))

        return

    def mult_vec_and_state_real(self, op_idx, x):
        # TODO Access the coeffs of the state data for obj
        # Sum everything after mult
        # TODO Check the signs
        s_real = np.array([i.real for i in self.states[op_idx].data])
        s_imag = np.array([i.imag for i in self.states[op_idx].data])
        vec_real = np.array(self.whole_vec_real(op_idx, x))
        vec_imag = np.array(self.whole_vec_imag(op_idx, x))
        final_real = np.inner(s_real, vec_real) - np.inner(s_imag, vec_imag)
        final_imag = np.inner(s_real, vec_imag) + np.inner(s_imag, vec_real)
        return final_real**2 + final_imag**2

    def obj(self, x):
        # The final objective function
        return quicksum(self.mult_vec_and_state_real(i, x) for i in range(self.num_ops))

    def set_init(self, x):
        self.x0 = x.copy()
        return

    def find_init(self, method=None):
        if method == "zero":
            self.x0 = np.zeros(self.num_vars)
        # TODO Find a better initial point with brute-force heuristic
        else:
            x = np.random.random_sample(self.num_vars) * 2 - 1
            self.x0 = self.norm_vars(x)
        return

    def norm_vars(self, x, inplace=False):
        """Normalize the variable vector to make sure the combined vector is unit length."""
        x_unit = []
        for j in range(0, self.num_vars, self.num_vars_per_op):
            x_slice = x[j : j + self.num_vars_per_op]
            r = np.linalg.norm(x_slice)
            x_slice_unit = np.multiply(x_slice, 1 / r)
            x_unit.extend(x_slice_unit)
        return x_unit

    def solve(self, method="COBYQA", **options):
        # set_nonlinear_objective(self.model, self.obj(self.x), "minimize")
        new_obj = self.model.addVar(lb=0, obj=1)
        # new_obj = self.model.addVar(lb=1e-09, obj=1)

        self.model.addCons(self.obj(self.x) <= new_obj)
        self.model.setMinimize()
        # self.model.hideOutput()
        model = self.model
        #self.model.optimize()
        model.optimize()
        x = model.data
        print(x)
        # TODO
        print(model.getNSolsFound())
        # print(model.getNCountedSols())
        print(model.getSols())
        # TODO Print both the solutions
        # TODO Conflict analysis

        return

    def expand_solve(self):
        states = self.states
        # Extend states
        for i in range(self.num_states):
            states[i] = states[i].expand(Statevector([1, 0]))
            print(states[i].data)
        self.__init__(num_qubits=self.num_qubits + 1, num_states=self.num_states)
        self.set_states(states=states)
        return

    def solve_feas(self, method="COBYQA", **options):
        # Solve feasibility problem
        return

    def is_lin_indep(self):
        """Check the vectors are linearly independent."""
        l = []
        for i in range(self.num_states):
            l.append(self.states[i].data)
        m = np.array(l)
        rank = np.linalg.matrix_rank(m)
        print("Rank =", rank)
        print("#states =", self.num_states)

    def verify(self):
        if type(self.x) is int:
            return False

        print("Optimal value:", self.model.getObjVal())
        l = []
        for i in range(self.num_vars):
            l.append(self.model.getVal(self.model.data[i]))
        print(np.array(l))
        self.x = np.array(l)

        print("My objective =", self.obj(self.x))

        final_ops = self.calc_final_ops()
        almost = 0
        success = 0
        for o1, o2 in combinations(final_ops, 2):
            vdot = np.round(np.vdot(o1, o2), 4)
            if 0.3 < np.abs(vdot) <= 0.5:
                almost += 1
            elif np.abs(vdot) <= 0.3:
                success += 1
            print("vdot:", vdot)
        for i in range(self.num_ops):
            print("unit:", round(self.op_con(i, self.x), 4))
        num_pairs = int(self.num_ops * (self.num_ops - 1) / 2)
        if success == num_pairs:
            print("Success (all <= 0.3)")
            return True
        elif almost + success == num_pairs:
            print("Almost (all <= 0.5)")
            print(f"almost (<= 0.5): {almost}, success (<= 0.3): {success}")
            return False
        else:
            return False

    def calc_final_ops(self):
        final_ops = []
        for op_idx in range(self.num_ops):
            vec = 0
            for j in range(self.num_basis):
                basis = self.null_spaces[op_idx][:, j]
                idx = op_idx * self.num_vars_per_op + 2 * j
                vec += basis * (self.x[idx] + self.x[idx + 1] * 1j)
            final_ops.append(vec)
        return final_ops

    @staticmethod
    def test_lin_indep():
        np.set_printoptions(precision=4)
        prob = NullSpaceSearchProblem(num_qubits=2, num_states=3)
        prob.set_states()
        prob.is_lin_indep()

    @staticmethod
    def test():
        np.set_printoptions(precision=4)
        prob = NullSpaceSearchProblem(num_qubits=2, num_states=3)
        prob.set_states()
        # print(prob.states)
        prob.find_null_spaces()
        
        prob.build_cons()
        prob.find_init()
        prob.solve()
        # prob.x = prob.norm_vars(prob.x)
        prob.verify()

        return

    @staticmethod
    def test_brute():
        np.set_printoptions(precision=4)
        prob = NullSpaceSearchProblem(num_qubits=2, num_states=3)
        prob.set_states()
        # print(prob.states)
        prob.find_null_spaces()
        prob.build_cons()
        num_samples = 0
        while not prob.verify():
            num_samples += 1
            print(num_samples)
            x = np.random.random_sample(prob.num_vars) * 2 - 1
            # prob.set_init(x=x_unit)
            # prob.solve()
            prob.x = prob.norm_vars(x)
        print(num_samples)
        print(prob.x)

        return

    def solve_and_lock(self):
        # Fix one of the vectors and solve with null spaces again
        # TODO
        return

    @staticmethod
    def test_brute_multistart():
        np.set_printoptions(precision=4)
        prob = NullSpaceSearchProblem(num_qubits=2, num_states=3)
        prob.set_states()
        # print(prob.states)
        prob.find_null_spaces()
        prob.build_cons()
        num_samples = 0
        stop = False
        while not stop:
            num_samples += 1
            print(num_samples)
            x = np.random.random_sample(prob.num_vars) * 2 - 1
            x_unit = []
            for j in range(0, prob.num_vars, prob.num_vars_per_op):
                x_slice = x[j : j + prob.num_vars_per_op]
                r = np.linalg.norm(x_slice)
                x_slice_unit = np.multiply(x_slice, 1 / r)
                x_unit.extend(x_slice_unit)
            # TODO Normalize by pairs, not all of them!
            prob.set_init(x=x_unit)
            prob.solve()
            # prob.x = np.array(x_unit)
            prob.x = prob.norm_vars(prob.x)
            stop = prob.verify()
        print(num_samples)
        print(prob.x)

        return

    @staticmethod
    def test_expand_solve():
        np.set_printoptions(precision=4)
        prob = NullSpaceSearchProblem(num_qubits=2, num_states=3)
        prob.set_states()
        prob.expand_solve()
        # print(prob.states)
        prob.find_null_spaces()
        prob.build_cons()
        prob.find_init()
        prob.solve()
        prob.verify()
        return

    # prods = []

    # TODO define callback
    def callbackF(Xi):
        global Nfeval
        # print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], rosen(Xi))
        Nfeval += 1


def ssp(V, E):
    """ssp -- model for the stable set problem
    Parameters:
        - V: set/list of nodes in the graph
        - E: set/list of edges in the graph
    Returns a model, ready to be solved.
    """
    model = Model("ssp")

    x = {}
    # TODO add lb, ub
    # TODO quicksum
    for i in V:
        x[i] = model.addVar(vtype="C", name="x(%s)" % i)

    for i, j in E:
        model.addCons(x[i] + x[j] <= 1, "Edge(%s,%s)" % (i, j))

    
    # model.data = x, y, z, v # Vars
    return model


if __name__ == "__main__":
    # NullSpaceSearchProblem.test_brute_multistart()
    # NullSpaceSearchProblem.test_brute()
    NullSpaceSearchProblem.test_expand_solve()
    
    # model.setObjective(quicksum(x[i] for i in V), "maximize")
    # model.data = x
    # x = model.data
    # # x, y, z, v = model.data
    # print("Maximum stable set:")
    # print([i for i in V if model.getVal(x[i]) > 0.5])
    #
