import sys

sys.path.append("../")
import logging

from flow.solve_mix import *
from flow.build_circuits import *
from utils.handy_states import *
from temp.get_random_seeds import *


def build_benchmark_circuit_coh_asymm(num_qubits, qsd_method="MED"):
    state_dict = coh_asymm_small(num_qubits=num_qubits)
    num_states = state_dict["num_states"]
    qsd_problem = ProblemSpec(
        num_qubits=state_dict["num_qubits"], num_states=state_dict["num_states"]
    )
    qsd_problem.set_states(
        state_type="densitymatrix",
        states=state_dict["dense_mat"],
        overwrite=True,
    )
    if qsd_method == "MED":
        eps = 1e-8
        cvxpy_settings = {"solver": cp.MOSEK, "verbose": False, "eps": eps}
        cvxpy_problem = med_problem(qsd_problem)
        cvxpy_problem.solve(**cvxpy_settings)
        vars = cvxpy_problem.variables()
        povm = [var.value for var in vars]
    elif qsd_method == "OptUQSD":
        eps = 1e-8
        cvxpy_settings = {"solver": cp.MOSEK, "verbose": False, "eps": eps}
        cvxpy_problem = med_problem(qsd_problem)
        cvxpy_problem.solve(**cvxpy_settings)
        vars = cvxpy_problem.variables()
        povm = [var.value for var in vars]

    povm_vectors, _ = povm_to_rank1_vectors(povm, threshold=0)
    povm_circuit = POVMCircuit(povm_vectors=povm_vectors)
    print(len(povm_vectors))
    print(len(povm_circuit.povm_vectors))
    povm_circuit.num_qubits = num_qubits
    povm_circuit.num_amps = 2**num_qubits
    povm_circuit.case_id = (
        f"circuits//coherent//coh_asymm_q{num_qubits}_n{num_states}_med_fullpovm_csd"
    )
    fullpovm_csd_qc = povm_circuit.build_circuit(scheme="csd")
    povm_circuit.case_id = (
        f"circuits//coherent//coh_asymm_q{num_qubits}_n{num_states}_med_fullpovm_ccd"
    )
    fullpovm_ccd_qc = povm_circuit.build_circuit(scheme="ccd")
    del povm_circuit

    reduced_povm_vectors, _ = povm_to_rank1_vectors(
        povm, threshold=1e-4
    )
    reduced_povm_circuit = POVMCircuit(povm_vectors=reduced_povm_vectors)

    print(verify_povm(reduced_povm_vectors))
    print(len(reduced_povm_vectors))
    print(len(reduced_povm_circuit.povm_vectors))
    reduced_povm_circuit.num_qubits = num_qubits
    reduced_povm_circuit.num_amps = 2**num_qubits
    reduced_povm_circuit.case_id = f"circuits//coherent//coh_asymm_q{num_qubits}_n{num_states}_med_reducedpovm_csd"

    reduced_med_qc = reduced_povm_circuit.build_circuit(scheme="csd")
    reduced_povm_circuit.case_id = f"circuits//coherent//coh_asymm_q{num_qubits}_n{num_states}_med_reducedpovm_ccd"
    reduced_med_ccd_qc = reduced_povm_circuit.build_circuit(scheme="ccd")


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"20250912_benchmark_gen.log",
        filemode="a",
        format="{asctime} {levelname} {filename}:{lineno}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=logging.INFO,  # Qiskit dumps too many DEBUG messages
        encoding="utf-8",
    )

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("PIL.PngImagePlugin").disabled = True
    logging.getLogger("matplotlib.mathtext").disabled = True
    logging.getLogger("qiskit.passmanager.base_tasks").disabled = True
    logger = logging.getLogger(__name__)
    # build_benchmark_circuit_coh_symm(2)
    # build_benchmark_circuit_coh_symm(3)
    # build_benchmark_circuit_coh_symm(4)
    # build_benchmark_circuit_coh_symm(5)
    # build_benchmark_circuit_coh_symm(6)
    
    build_benchmark_circuit_coh_asymm(2)
    build_benchmark_circuit_coh_asymm(3)
    build_benchmark_circuit_coh_asymm(4)
    build_benchmark_circuit_coh_asymm(5)
    build_benchmark_circuit_coh_asymm(6)
