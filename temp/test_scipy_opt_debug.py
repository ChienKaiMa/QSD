import numpy as np
from itertools import combinations
from functools import partial
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize
from scipy.linalg import null_space
import cobyqa
from qiskit.quantum_info import random_statevector, Statevector

# https://stackoverflow.com/questions/55132107/scipy-fitting-with-parameters-in-a-vector
from operator import add

import cvxpy as cp


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
        self.x = 0

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

            null_s = null_space(np.conj(np.array(A)))
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

    # def con(self, i, x):
    #     return x[2 * i] ** 2 + x[2 * i + 1] ** 2

    def build_cons(self):
        self.constraints = []

        for i, j in combinations(list(range(self.num_ops)), 2):
            self.constraints.append(
                NonlinearConstraint(partial(self.mult_vec_and_vec_real, i, j), 0, 0)
            )
            self.constraints.append(
                NonlinearConstraint(partial(self.mult_vec_and_vec_imag, i, j), 0, 0)
            )

        for i in range(self.num_ops):
            self.constraints.append(NonlinearConstraint(partial(self.op_con, i), 1, 1))
            # constraints.append(NonlinearConstraint(partial(op_con, i), 0.85, 1))

        # TODO this one is not actually required?
        # for i in range(self.num_vars // 2):
        #     self.constraints.append(NonlinearConstraint(partial(self.con, i), 0, 1))

        # self.constraints = constraints.copy()
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
        s = 0
        for i in range(self.num_ops):
            s += self.mult_vec_and_state_real(i, x)
        return -s

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
        assert type(self.x0) == list
        if method == "COBYQA":
            result: OptimizeResult = cobyqa.minimize(
                fun=self.obj,
                x0=self.x0,
                constraints=self.constraints,
                options={
                    "target": 0,
                    "maxfev": 30000,
                    # "disp": True,
                    # "radius_init": 0.6,
                },
                decrease_radius_factor=0.3,
                # increase_radius_factor=4.0,
            )
        elif method == "SLSQP":
            result = minimize(
                fun=self.obj,
                x0=self.x0,
                constraints=self.constraints,
                method="SLSQP",
                options={
                    "maxiter": 40000,
                    "ftol": 1e-10,
                },
            )
        elif method == "Nelder-Mead":
            result = minimize(
                fun=self.obj,
                x0=self.x0,
                constraints=self.constraints,
                method="Nelder-Mead",
                options={
                    "maxiter": 40000,
                },
            )
        print(result)
        self.x = result["x"]
        print(self.x)
        return

    def expand_solve(self):
        # Extend states
        states = self.states
        for i in range(self.num_states):
            states[i] = states[i].expand(Statevector([1, 0]))
            # print(states[i].data)
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

        final_ops = self.calc_final_ops()
        almost = 0
        success = 0
        for op in final_ops:
            for i in range(self.num_states):
                vd = np.round(np.vdot(self.states[i].data, op), 4)
                print("vdot:", vd)
                # d = np.round(np.dot(self.states[i].data, op), 4)
                # print("dot:", d)
        
        for o1, o2 in combinations(final_ops, 2):
            vd = np.round(np.vdot(o1, o2), 4)
            if 0.3 < np.abs(vd) <= 0.5:
                almost += 1
            elif np.abs(vd) <= 0.3:
                success += 1
            print("vdot:", vd)

        # Check objective function
        a = 0
        for i in range(self.num_ops):
            # vdot or matmul?
            # a += (np.abs(np.vdot(self.states[i].data, final_ops[i])) ** 2)
            print(self.states[i].data)
            a += np.linalg.norm(np.vdot(final_ops[i], self.states[i].data)) ** 2
        for i in range(self.num_ops):
            print(final_ops[i])
        print("a =", a)

        for i in range(self.num_ops):
            print("unit:", np.linalg.norm(final_ops[i]))
            # print("unit:", round(self.op_con(i, self.x), 4))
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
        prob.verify()

        return


    @staticmethod
    def test_expand_solve():
        np.set_printoptions(precision=4)
        prob = NullSpaceSearchProblem(num_qubits=2, num_states=3)
        prob.set_states()
        prob.expand_solve()
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
            # prob.solve(method="SLSQP")
            prob.solve()
            # prob.x = np.array(x_unit)
            prob.x = prob.norm_vars(prob.x)
            stop = prob.verify()
        print(num_samples)
        print(prob.x)
        return

    # prods = []

    # TODO define callback
    def callbackF(Xi):
        global Nfeval
        # print '{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], rosen(Xi))
        Nfeval += 1

    def apply_Eldar(self, prior_prob=None, min_prob=0):
        """Apply the method in Eldar's paper in 2003."""
        np.set_printoptions(precision=4)
        n = self.num_states
        if prior_prob is None:
            prior_prob = np.ones(n) * (-1 / n)

        A = []
        for s in self.states:
            A.append(s)
        psi = np.transpose(A)
        # print(psi)
        recip_psi = np.matmul(psi, np.linalg.inv(np.matmul(np.matrix.getH(psi), psi)))
        # Round nearzero value
        recip_psi = recip_psi.round(15)
        # print(recip_psi)
        recip_psi = recip_psi.T
        # q1 = np.outer(recip_psi[0], recip_psi[0])
        # print(q1)

        # Measurement operators
        q = []
        for i in range(n):
            q.append(np.outer(recip_psi[i], recip_psi[i]).round(1))
        q = np.array(q)

        I = np.identity(self.num_amps)
        p = cp.Variable(n)

        objective = cp.Minimize(1 + cp.sum(prior_prob @ p))

        constraints = []
        assert min_prob >= 0
        for i in range(n):
            constraints.append(min_prob <= p[i])
            constraints.append(p[i] <= 1)

        expr = I
        for i in range(n):
            expr = expr - p[i] * q[i]
        constraints.append(expr >> 0)  # Matrix inequality uses >>

        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        print("Result =", result.round(3))
        # An acceptable optimal solution
        sol = p.value.round(4)
        print("Solution =", sol)
        # Positive semidefinite
        pi1 = I
        for i in range(n):
            pi1 = pi1 - sol[i] * q[i]

        for i in range(n):
            print(np.sqrt(sol[i]) * recip_psi[i])
            print(np.linalg.norm(np.sqrt(sol[i]) * recip_psi[i]))
        # print(pi1.round(5))
        # Wrong answer if we over postprocess the solution
        # sol_overround = p.value.round(2)
        # print("Overprocessed solution =", sol_overround)
        # pi1_overround = (
        #     I
        #     - sol_overround[0] * q[0]
        #     - sol_overround[1] * q[1]
        #     - sol_overround[2] * q[2]
        # )  # Not positive semidefinite
        # print(pi1_overround.round(5))

        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
        # print(constraints[0].dual_value)
        return

    @staticmethod
    def test_Eldar():
        prob = NullSpaceSearchProblem(num_qubits=2, num_states=3)
        prob.set_states()
        prob.apply_Eldar()
        return


if __name__ == "__main__":
    NullSpaceSearchProblem.test()
