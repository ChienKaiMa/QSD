import sys
import time
import tracemalloc
import logging
from argparse import *

sys.path.append("../")
sys.path.append("./")
from temp.get_random_seeds import *
from qiskit.quantum_info import (
    random_statevector,
    random_density_matrix,
)


class ProblemSpec:
    def __init__(
        self,
        num_qubits: int,
        num_states: int,
        seed: int | np.random.Generator | None = 42,
        case_id: str = "",
        prior_prob: list[int] | None = None,
        state_type="statevector",
    ):
        logger = logging.getLogger(__name__)
        assert num_qubits > 0
        assert num_states > 1
        self.num_qubits = num_qubits
        self.num_states = num_states
        self.seed = seed
        if case_id == "":
            self.case_id = f"q{num_qubits}_n{num_states}_s{seed}"
            logger.info(f"The case id is set to {self.case_id}")
        else:
            self.case_id = case_id
        self.num_amps = 2**num_qubits
        self.num_ops = num_states
        self.states = []
        self.constraints = []
        self.prior_prob = prior_prob
        self.state_type = state_type

    @staticmethod
    def gen_states(
        num_qubits: int,
        num_states: int,
        seeds: list[int] = [],
        state_type="statevector",
        **kwargs,
    ):
        logger = logging.getLogger(__name__)
        assert num_qubits > 0
        assert num_states > 1
        # TODO Sparse state generation
        if len(seeds):
            if len(seeds) != num_states:
                logger.error(
                    f"The number of provided seeds ({len(seeds)}) does not match the number of states ({num_states})"
                )
                return
            if len(seeds) != len(set(seeds)):
                logger.error(f"There are duplicate seeds")
                return
            logger.info("Use the seeds from the keyword arguments")
        else:
            logger.info(f"Use range({num_states}) as the seeds")
            seeds = list(range(num_states))

        if state_type == "statevector":
            states = [
                random_statevector(
                    2**num_qubits,
                    seed=seeds[_],
                )
                for _ in range(num_states)
            ]
        elif state_type == "densitymatrix":
            states = [
                random_density_matrix(
                    dims=2**num_qubits,
                    rank=2,  # TODO rank
                    method="Hilbert-Schmidt",  # TODO method
                    seed=seeds[_],
                )
                for _ in range(num_states)
            ]
        logger.info(
            f"{num_states} random {num_qubits}-qubit states ({state_type}) are generated"
        )
        return states

    def set_states(self, state_type, states=None, overwrite=False):
        logger = logging.getLogger(__name__)
        if states is None:
            logger.info("No states are provided. The states will be generated.")
            self.states = self.gen_states(
                num_qubits=self.num_qubits,
                num_states=self.num_states,
                state_type=self.state_type,
            )
        elif self.states != [] and not overwrite:
            logger.warning(
                "The states are already set. This method call will do nothing."
            )
            return
        elif overwrite:
            # TODO assert type correspond to data
            self.states = states.copy()
            self.state_type = state_type
        return

    pass
