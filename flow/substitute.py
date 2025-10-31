# Substitute rotation gates
import sys
import time
import tracemalloc
import logging
from argparse import *

from qiskit import QuantumCircuit
import qiskit.synthesis
import qiskit.transpiler


def agsub(qc: QuantumCircuit, ratio=0.1):
    """Aggressive substitution

    TODO Some properties I may need
    qc.count_ops()
    qc.num_connected_components()
    qc.remove_final_measurements()
    qc.num_unitary_factors()
    qc.num_parameters
    qc.num_nonlocal_gates()
    from qiskit.circuit.library import UGate
    UGate().control()
    """
    # Count rotation gates
    from qiskit import transpile

    qc_d = transpile(qc, basis_gates=["rx", "ry", "rz", "cx"])
    print(qc_d.count_ops())
    print(qc_d.depth())
    # Couldn't transpile for qc_iso_q3_n7_s3_no_backend.qasm
    # qc_d_2 = transpile(qc, basis_gates=["rx", "sx", "cx"])
    # qc_d_2 = transpile(qc, basis_gates=["rx", "sx", "ecr"])
    # print(qc_d_2.count_ops())
    # print(qc_d_2.depth())
    # TODO Start your game!
    # for gate in qc_d:
    #     print()

    return


def modsub(
    qc,
):
    """Moderate substitution"""
    return


def hybridsub(qc):
    """Hybrid substitution"""
    return


def quick_noisy_sim():
    """Check how noise affects the discrimination performance

    Omit measurement errors
    """
    return


# Structural resynthesis


if __name__ == "__main__":
    # TODO
    # Simple tests or solver comparison?
    parser = ArgumentParser()
    parser.add_argument("-q", "--qasm")
    # parser.add_argument("-n", "--nstates", default=3)
    # parser.add_argument("-s", "--seed", default=42)
    parser.add_argument("-r", "--ratio", default=0.1)
    args = parser.parse_args()
    agsub_ratio = args.ratio
    # nq = int(args.nqubits)
    # ns = int(args.nstates)
    # seed = int(args.seed)
    # case_id = f"q{nq}_n{ns}_s{seed}"
    qasm_name = args.qasm
    qc = QuantumCircuit.from_qasm_file(qasm_name)
    print(qc.count_ops())
    print(qc.depth())
    # TODO remove folder and .qasm from qasm_name
    logging.basicConfig(
        filename=f"opt_{qasm_name}.log",
        filemode="a",
        format="{asctime} {levelname} {filename}:{lineno}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
        level=logging.DEBUG,
        encoding="utf-8",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Start a new program")
    import qiskit
    from qiskit import transpile

    qc = transpile(qc, basis_gates=["rx", "ry", "rz", "cx"])
    print(int(qc.depth() * 0.05))
    qc_approx = transpile(
        circuits=qc,
        unitary_synthesis_method="aqc",
        unitary_synthesis_plugin_config={
            "network_layout": "cart",
            "connectivity_type": "star",
            "depth": int(qc.depth() * 0.05),
        },
    )
    print("Approx")
    print(qc_approx.count_ops())
    print(qc_approx.depth())
    # agsub(qc, ratio=agsub_ratio)
    # modsub(
    #     qc,
    # )
    ## # [:-15] Remove "no_backend.qasm" and add "ibm_brisbane.qasm"
    qc = QuantumCircuit.from_qasm_file(
        qasm_name[:-15] + "ibm_brisbane.qasm"
    )
    print(qc.count_ops())
    print(qc.depth())
