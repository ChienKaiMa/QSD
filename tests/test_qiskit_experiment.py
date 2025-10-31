from typing import Tuple, List, Optional, Sequence
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_experiments.framework import (
    BaseAnalysis,
    BaseExperiment,
    Options,
    ExperimentData,
    AnalysisResultData,
)
from exp_uqsd_med import *


class QSDExperiment(BaseExperiment):
    """Class for Quantum State Discrimination (QSD) experiments.

    Args:
        BaseExperiment (class): Abstract base class for experiments.
    """

    # https://qiskit-extensions.github.io/qiskit-experiments/tutorials/custom_experiment.html
    def __init__(
        self,
        physical_qubits: Sequence[int] = None,
        backend: Optional[Backend] = None,
        circuit_type="UQSD",
    ):
        """Initialize the experiment."""
        # TODO
        # Decide parameters
        self.circuit_type = circuit_type.upper()
        self.backend = backend
        ckts = self.circuits()
        if physical_qubits is None:
            # TODO
            # Check if self.backend is AerSimulator
            # ibmq_qasm_simulator is a trouble maker, please don't use it!
            physical_qubits = tuple(range(ckts[0].num_qubits))
        # TODO
        super().__init__(physical_qubits, analysis=CustomAnalysis(), backend=backend)
        # TODO
        # Check required parameters
        pass

    def circuits(self) -> List[QuantumCircuit]:
        # TODO 1
        ckts = []
        # if self.circuit_type == "UQSD":
        #     qc = experiment(
        #         num_qubit=num_qubit,
        #         inner_product=inner_product,
        #         method=method,
        #         initial_state=circuit_type,
        #         probability=probability,
        #     )
        # elif self.circuit_type == "MED":
        #     pass
        circuit = QuantumCircuit(2)
        circuit.h(1)
        ckts.append(circuit)

        return ckts

    def run(
        self,
        backend: Backend | None = None,
        analysis: BaseAnalysis | None = "default",
        timeout: float | None = None,
        **run_options,
    ) -> ExperimentData:
        return super().run(backend, analysis, timeout, **run_options)

    # def run_on_ibmq(self):
    #     return self.run()

    # def run_on_ionq(self):
    #
    #     from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    #
    #     target = backend.target
    #
    #     from qiskit import qasm2
    #
    #     circuits_ibm = []
    #     for circuit in circuits:
    #         # pm = generate_preset_pass_manager(target=target, optimization_level=3)
    #         # pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    #         # circuit_ibm = pm.run(circuit)
    #         circuit_ibm = transpile(circuit, backend=backend, optimization_level=3)
    #         print(f"{circuit_ibm.count_ops()}")
    #         print(f"Layers: {circuit_ibm.depth()}")
    #         # For Qiskit 0.46
    #         # circuit_clean = QuantumCircuit.from_qasm_str(circuit_ibm.qasm())
    #         circuit_clean = QuantumCircuit.from_qasm_str(qasm2.dumps(circuit_ibm))
    #         circuits_ibm.append(circuit_clean)
    #
    #     return self.run()

    def print_circuits(self):
        # circuit.draw(
        #     output="mpl",
        #     filename=f"./exp0/figures/"
        #     + "exp0"
        #     + f"_{experiment_type}_transpile"
        #     # + f"_{c0_bob:.3f}"
        #     # + f"_{n_qubit}_{seed}"
        #     + ".png",
        # )
        # TODO
        return

    # def _transpiled_circuits(self) -> List[QuantumCircuit]:
    #     from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


#
#     pm = generate_preset_pass_manager(
#         optimization_level=1,
#         target=self.backend.target,
#     )
#     isa_circuits = pm.run(self.circuits())
#     return isa_circuits
#     # return super()._transpiled_circuits()

#
# def set_transpile_options(self, **fields):
#     return super().set_transpile_options(**fields)


import matplotlib
from typing import Tuple, List
from qiskit_experiments.framework import (
    BaseAnalysis,
    Options,
    ExperimentData,
    AnalysisResultData,
)


class CustomAnalysis(BaseAnalysis):
    """Custom analysis class template."""

    @classmethod
    def _default_options(cls) -> Options:
        """Set default analysis options. Plotting is on by default."""

        options = super()._default_options()
        options.dummy_analysis_option = None
        options.plot = True
        options.ax = None
        return options

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Run the analysis."""

        # Process the data here

        analysis_results = [AnalysisResultData(name="dummy result", value=data)]
        figures = []
        if self.options.plot:
            figures.append(self._plot(data))
        return analysis_results, figures


if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel="ibm_quantum",
        instance="ibm-q-hub-ntu/ntu-internal/default",
        token="ee4393fc53fe3dc29ff3765f60a46fd50eb650536a98580a688346c7bf275049012a3bb969d448aa6fced96fc62bd05a565a60d7fdfee7da80bbc01a434d5aa8",
    )
    print(service.backends())

    backend = service.backend("ibm_sherbrooke")
    backend = service.backend("ibm_brisbane")
    # # sim_backend = AerSimulator()
    # sim_exp = QSDExperiment(backend=sim_backend)
    # print(sim_exp.transpile_options.__dict__)
    # circuits = sim_exp._transpiled_circuits()
    # # transpiled = transpile(sim_exp.circuits(), target=sim_exp.backend.target, initial_layout=(0,1))
    # transpiled = transpile(sim_exp.circuits(), target=sim_exp.backend.target)
# 
    # for circuit in transpiled:
    #     print(circuit)
    # sim_exp.run(backend=sim_backend)
    # # sim_exp.run(backend=sim_backend)
    # # print(sim_exp.job_info())
    # # ibmq_backend = service.backend("ibm_osaka")
    # # ibmq_exp = QSDExperiment(backend=ibmq_backend)

    # from qiskit_braket_provider import BraketProvider

    # ionq_backend = BraketProvider().get_backend("Aria 1")
    # ionq_exp = QSDExperiment(backend=ionq_backend)
