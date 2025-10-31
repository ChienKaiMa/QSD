import numpy as np
import scipy.linalg
from itertools import combinations
from functools import partial


from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize
from scipy.linalg import null_space
import cobyqa
from qiskit.quantum_info import (
    random_statevector,
    random_density_matrix,
    Statevector,
    DensityMatrix,
)
from qiskit.synthesis import qs_decomposition
from qiskit.circuit.library import Isometry

# copied from build_circuits.py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import qiskit
import qiskit_ibm_runtime
import qclib
import random
from datetime import datetime
import os
import time
import tracemalloc

from math import pi
from scipy.spatial.transform import Rotation as R
from scipy.linalg import qz

from qiskit import transpile
from qiskit import QuantumRegister, ClassicalRegister, AncillaRegister, QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.circuit import Parameter
from qiskit.providers.basic_provider import BasicProvider
from qiskit.primitives import Sampler
import qiskit.qasm2

from qiskit_aer import AerProvider

from qclib.isometry import decompose
from qclib.state_preparation import UCGInitialize, IsometryInitialize

# from qiskit.quantum_info import *
from get_random_seeds import *

# https://stackoverflow.com/questions/55132107/scipy-fitting-with-parameters-in-a-vector
from operator import add

import cvxpy as cp
import sys


# TODO Come up with a better class name
class StateDiscriminationProblem:
    def __init__(
        self, num_qubits, num_states, prior_prob=None, state_type="statevector"
    ):
        assert num_qubits > 0
        assert num_states > 1
        self.num_qubits = num_qubits
        self.num_states = num_states
        self.num_amps = 2**num_qubits
        self.num_ops = num_states
        # self.num_basis = 2**num_qubits - (num_states - 1)
        # self.num_vars_per_op = self.num_basis * 2
        # self.num_vars = self.num_ops * self.num_basis * 2
        # self.num_coeffs = self.num_vars * self.num_amps
        self.states = []
        self.null_spaces = []
        self.null_coeffs = []
        self.constraints = []
        self.prior_prob = prior_prob
        self.state_type = state_type
        self.x0 = 0
        self.x = 0

    @staticmethod
    def gen_states(
        num_qubits: int,
        num_states: int,
        seeds: list[int] = [],
        state_type="statevector",
        **kwargs,
    ):
        assert num_qubits > 0
        assert num_states > 1
        states = []
        # TODO Sparse option
        if len(seeds):
            # Setting seeds from the arguments
            # TODO Check duplicates in seeds
            if len(seeds) == num_states:
                if state_type == "statevector":
                    states = [
                        random_statevector(2**num_qubits, seed=seeds[_])
                        for _ in range(num_states)
                    ]
                elif state_type == "densitymatrix":
                    # TODO rank
                    states = [
                        random_density_matrix(
                            dims=2**num_qubits,
                            rank=2,
                            seed=seeds[_],
                            method="Hilbert-Schmidt",
                        )
                        for _ in range(num_states)
                    ]
            else:
                print(
                    f"The number of provided seeds ({len(seeds)}) does not match the number of states ({num_states})",
                    file=sys.stderr,
                )
        else:
            # Use range as seeds
            if state_type == "statevector":
                states = [
                    random_statevector(2**num_qubits, seed=_) for _ in range(num_states)
                ]
            elif state_type == "densitymatrix":
                states = [
                    random_density_matrix(dims=2**num_qubits, rank=2, seed=_)
                    for _ in range(num_states)
                ]
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
                state_type=self.state_type,
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
        final_real = np.inner(s_real, vec_real) + np.inner(s_imag, vec_imag)
        final_imag = np.inner(s_real, vec_imag) - np.inner(s_imag, vec_real)
        return final_real**2 + final_imag**2

    def obj(self, x):
        # The final objective function
        s = 0
        for i in range(self.num_ops):
            s += self.prior_prob[i] * self.mult_vec_and_state_real(i, x)
        return -s

    def set_init(self, x):
        self.x0 = x.copy()
        return

    def find_init(self, method=None):
        if method == "zero":
            self.x0 = np.zeros(self.num_vars)
        # TODO Find a better initial point with brute-force heuristic
        else:
            # method == "rand_norm"
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

        if self.prior_prob is None:
            self.prior_prob = np.ones(self.num_states) * (1 / self.num_states)
        if method == "COBYQA":
            result: OptimizeResult = cobyqa.minimize(
                fun=self.obj,
                x0=self.x0,
                constraints=self.constraints,
                options={
                    "target": -1,
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
            a += (
                self.prior_prob[i]
                * np.linalg.norm(np.vdot(final_ops[i], self.states[i].data)) ** 2
            )
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
        prob = StateDiscriminationProblem(num_qubits=2, num_states=3)
        prob.set_states()
        prob.is_lin_indep()

    @staticmethod
    def test():
        np.set_printoptions(precision=4)
        prob = StateDiscriminationProblem(num_qubits=2, num_states=3)
        prob.set_states()
        # print(prob.states)
        prob.find_null_spaces()
        prob.build_cons()
        prob.find_init()
        prob.solve(method="SLSQP")
        prob.x = prob.norm_vars(prob.x)
        prob.verify()

        return

    @staticmethod
    def test_brute():
        # Skip any solver and check random assignments
        np.set_printoptions(precision=4)
        prob = StateDiscriminationProblem(num_qubits=2, num_states=3)
        prob.set_states()
        # print(prob.states)
        prob.find_null_spaces()
        prob.build_cons()
        num_samples = 0
        while not prob.verify():
            num_samples += 1
            print(num_samples)
            x = np.random.random_sample(prob.num_vars) * 2 - 1
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
        # Solve with random starting points
        np.set_printoptions(precision=4)
        prob = StateDiscriminationProblem(num_qubits=2, num_states=3)
        prob.set_states()
        # print(prob.states)
        prob.find_null_spaces()
        prob.build_cons()
        num_samples = 0
        stop = False
        while not stop:
            num_samples += 1
            print(num_samples)
            prob.find_init()
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
        prob = StateDiscriminationProblem(num_qubits=2, num_states=3)
        prob.set_states()
        prob.expand_solve()
        prob.expand_solve()
        prob.find_null_spaces()
        prob.build_cons()
        num_samples = 0
        stop = False
        while not stop:
            num_samples += 1
            print(num_samples)
            prob.find_init()
            prob.solve(method="SLSQP")
            # prob.solve()
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
        prob = StateDiscriminationProblem(num_qubits=2, num_states=3)
        prob.set_states()
        prob.apply_Eldar()
        return

    def apply_Eldar_mix(self, prior_prob=None, p_I=0, min_prob=0):
        """Apply the method in Eldar's paper in 2004. SIM
        p_I: The predefined portion of inconclusive results. [0, 1)
        """
        np.set_printoptions(precision=4)
        n = self.num_states
        if prior_prob is None:
            prior_prob = np.ones(n) * (1 / n)

        ## A = []
        ## for s in self.states:
        ##     A.append(s)
        ## psi = np.transpose(A)
        ## # print(psi)
        ## recip_psi = np.matmul(psi, np.linalg.inv(np.matmul(np.matrix.getH(psi), psi)))
        ## # Round nearzero value
        ## recip_psi = recip_psi.round(15)
        ## # print(recip_psi)
        ## recip_psi = recip_psi.T
        ## # q1 = np.outer(recip_psi[0], recip_psi[0])
        ## # print(q1)

        # Measurement operators
        ## q = []
        ## for i in range(n):
        ##     q.append(np.outer(recip_psi[i], recip_psi[i]).round(1))
        ## q = np.array(q)

        I = np.identity(self.num_amps)
        X = cp.Variable(shape=(self.num_amps, self.num_amps), hermitian=True)
        delta_scalar = cp.Variable(1, name="d")
        beta = p_I  # Follow the naming in the paper

        objective = cp.Minimize(cp.trace(X) - delta_scalar * beta)

        # TODO [Priority: Low] add assertions
        Delta = np.sum(
            [np.multiply(prior_prob[i], self.states[i].data) for i in range(n)]
        )

        # Matrix inequality uses >>
        constraints = []
        for i in range(n):
            constraints.append(X - np.multiply(prior_prob[i], self.states[i].data) >> 0)
        constraints.append(X - cp.multiply(delta_scalar, Delta) >> 0)
        constraints.append(X >> 0)
        # TODO Add different constraint configurations
        # epsilon-oriented

        prob = cp.Problem(objective, constraints)
        # Solver options: The precision has to be 1e-16 ~ 1e-20 to be enough for SCS...
        ## result = prob.solve(solver=cp.CLARABEL)\
        ## No options for precision, and it couldn't find null vector
        ## result = prob.solve(solver=cp.CVXOPT, feastol=1e-10)
        ## CVXOPT failed when feastol is small (Not sure root cause)
        ## result = prob.solve(solver=cp.SCS, eps=1e-15)
        ## "eps > 1e-16" couldn't find null vector for operator
        result = prob.solve(solver=cp.SCS, eps=1e-20)
        print("Result =", result)

        # Please don't round X_sol
        X_sol = X.value
        delta_sol = delta_scalar.value
        print("Solution for X =")
        print(X_sol)
        print("Solution for delta =", delta_sol)

        # Find measurement operator
        povm = []
        for i in range(n):
            op = X_sol - np.multiply(prior_prob[i], self.states[i].data)
            # The precision here (rcond) also matters
            ## if the answer is not found, check the matrix and try a larger rcond
            ns = null_space(op, rcond=1e-7)
            # print(op)
            # TODO reshape ns
            # print(ns)
            # TODO
            povm.append(ns[:, 0])
            # res = np.all(np.linalg.eigvals(op) >= 0)
            # print(np.linalg.eigvals(op))
            # print(res)
        # The inconclusive measurement operator
        if beta == 0:
            print("Info: The inconclusive measurement is disabled.")
        # else:
        #     self.num_ops += 1
        #     op = X_sol - np.multiply(delta_sol, Delta)
        #     # The precision here (rcond) also matters
        #     ## if the answer is not found, check the matrix and try a larger rcond
        #     ns = null_space(op, rcond=1e-10)
        #     print(ns)
        #     povm.append(ns[:, 0])
        
        # 2024/10/30 Obtain PVM
        # The error measurement operator
        # if beta == 0:
        last_op = np.eye(self.num_amps, dtype='complex128')
        for m in povm:
            # Add "None" to transpose
            # https://stackoverflow.com/a/11885718/13518808
            op = np.multiply(m[None].T.conj(), m)
            last_op -= op
            # print(m)
        u, s, v = np.linalg.svd(last_op, hermitian=True)
        last_povm = u[:, 0] * np.sqrt(s[0])
        # print(last_povm)
        povm.append(last_povm.conj())
        
        # TODO
        # Verify solution
        # Positive semidefinite

        ## for i in range(n):
        ##     print(np.sqrt(sol[i]) * recip_psi[i])
        ##     print(np.linalg.norm(np.sqrt(sol[i]) * recip_psi[i]))
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

        isometry = self.naimark(povm)
        print(isometry)
        ## Isometry with Qiskit
        ## iso = Isometry(
        ##     isometry,
        ##     num_ancillas_zero=0,
        ##     num_ancillas_dirty=0,
        ## )
        ## qc_iso = QuantumCircuit(3)
        ## qc_iso.append(iso, [0, 1, 2])

        # 2024/10/30
        # csd works, while ccd doesn't
        qc_iso = decompose(isometry, scheme="csd").inverse()
        # qc_iso = decompose(isometry, scheme="ccd").inverse()

        service = qiskit_ibm_runtime.QiskitRuntimeService(
            channel="ibm_quantum",
            # instance="ibm-q/open/main",
            instance="ibm-q-hub-ntu/ntu-internal/default",
            token="e421d41292d0977e88ca2900d333e6b6789377af70e1923ba067e97afb929b2da3cd64bba701d1519067002f9c1fabe1e55c47a5539b12d8ec55b85864f6092d",
        )

        # Transpile first without the backend to avoid strange errors
        # qclib -> qiskit
        qc_iso = transpile(qc_iso)
        print(qc_iso.decompose(reps=2).count_ops())
        print("Depth,", qc_iso.decompose(reps=2).depth())
    
        # Transpile with the backend
        ibm_backend = service.backend("ibm_brisbane")
        qc_iso = transpile(qc_iso, backend=ibm_backend)
        print(qc_iso.count_ops())
        print("Depth,", qc_iso.depth())

        qiskit.qasm2.dump(
            qc_iso,
            f"qc_iso_q{self.num_qubits}_n{self.num_states}_noseed.qasm",
        )
        # t1_end = time.process_time()

        return

    def basis_extend(self, i, dims):
        a = np.zeros(dims)
        a[i] = 1
        # https://stackoverflow.com/questions/11885503/numpy-transpose-of-1d-array-not-giving-expected-result
        # print(np.matrix(a).T)
        return np.matrix(a).T

    def naimark(self, basis):
        V = sum(
            [
                np.multiply(
                    # self.basis_extend(i, self.num_amps * 2), np.matrix.getH(basis[i])
                    self.basis_extend(i, self.num_amps * 2), basis[i]
                )
                for i in range(len(basis))
            ]
        )
        return V
        ## print(self.num_ops)
        ## print(V)
        ## print("getH")
        ## mat_conj = np.matrix.getH(V)
        ## print(mat_conj)
        ## print(mat_conj.shape)
        ## print("pinv")
        ## mat_pinv = np.linalg.pinv(mat_conj)
        ## print(mat_pinv)
        ## print(mat_pinv.shape)
        ## ans = [np.matmul(mat_pinv, basis[i]) for i in range(self.num_ops)]
        ## print("ans")
        ## print(ans)
        ## return ans

    @staticmethod
    def test_obj(size=2):
        # Predefined vectors
        a = [0.5 + 0.2j, 0.3 + 0.4j, 0.5 + 0.2j, 0.5 + 0.2j]
        b = [0.9 - 0.3j, 0.1 + 0.4j, 0.2 + 0.4j, 0.5 + 0.2j]
        x = [1, 2, 3, 4]
        states = []
        states.append(a)
        states.append(b)
        prob = StateDiscriminationProblem(2, 2)
        prob.set_states(states)
        res = prob.mult_vec_and_state_real()
        print(res)
        return


if __name__ == "__main__":
    qsdp = StateDiscriminationProblem(2, 3)
    s = qsdp.gen_states(2, 3, state_type="densitymatrix")
    qsdp.set_states(s)
    # qsdp.apply_Eldar_mix(p_I=0)
    # qsdp.apply_Eldar_mix(p_I=0.1)
    # qsdp.apply_Eldar_mix(p_I=0) # OK
    qsdp.apply_Eldar_mix(p_I=0.1)
    # NullSpaceSearchProblem.test_brute_multistart()
    # NullSpaceSearchProblem.test_brute()
    # StateDiscriminationProblem.test_expand_solve()
    # NullSpaceSearchProblem.test_Eldar()
    # NullSpaceSearchProblem.test_obj()
    # denseList = []
    # qs_decomposition()
    # matA = []
    # matADag = []


def binaryCode(a):

    return


def grayCode(a):
    # Add an extra wrapper in case we want to change the dependencies
    import graycode

    return graycode.tc_to_gray_code(a)
    # code = graycode.gen_gray_codes(3)
    # return code[a]


def mab(k, a, b):
    return 2 ** (-k) * (-1) ** (b & grayCode(a))


def phi_a(k, a, theta):
    l = [mab(k, a, b) * theta[b] for b in range(2**k)]
    return sum(l)


def prepPureEnsemble(probList: list[float], svList: list[Statevector]):
    """
    probList: List[], the a priori probabilities
    """

    return


def prepMixedEnsemble(mixList: list[DensityMatrix]):

    return


def uniformCtrlSingleRot(phi: float, theta: float):
    # circuit.ry(theta)
    # circuit.rz(phi)
    return


def uniformCtrlRot(numQubits: int, thetaList: list[float]):
    """Uniformly controlled rotation gate
    numQubits: int, = k-fold
    """
    phiList = [sum()]
    return


def prepDenseByMatrixElements(mat):
    # Density matrix state preparation in terms of matrix elements in Bo-Hung's paper
    # mat is probably a numpy array
    c = scipy.linalg.cholesky(mat, overwrite_a=False, lower=True)

    # Explore sparsity?
    # Remove all-zero columns
    # Reference:
    # https://www.geeksforgeeks.org/how-to-remove-array-rows-that-contain-only-0-using-numpy/
    c = c[~np.all(c == 0, axis=0)]
    # Method from IV.A
    # Mixture of pure state

    return


def prettyGoodMeas(probList, opList):
    #
    return
