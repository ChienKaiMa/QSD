import numpy as np
from temp.get_random_seeds import *
import logging
from qiskit.quantum_info import (
    Statevector,
    random_statevector,
    random_density_matrix,
    DensityMatrix,
)

# random generators

# prepareInput


def _get_valid_seeds(seeds, num_states):
    logger = logging.getLogger(__name__)
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


def random_pure_states(
    num_qubits: int,
    num_states: int,
    seeds: list[int] = [],
    # state_type="statevector",
    **kwargs,
) -> list[Statevector]:
    # TODO Sparse state generation
    assert num_qubits > 0
    assert num_states > 1
    _get_valid_seeds(seeds=seeds, num_states=num_states)

    states = [
        random_statevector(
            2**num_qubits,
            seed=seeds[_],
        )
        for _ in range(num_states)
    ]

    logger = logging.getLogger(__name__)
    logger.info(
        f"{num_states} random {num_qubits}-qubit states (Statevector) are generated"
    )
    return states


def random_density_matrices(
    num_qubits: int,
    num_states: int,
    seeds: list[int] = [],
    # state_type="statevector",
    **kwargs,
):
    # TODO Sparse state generation
    assert num_qubits > 0
    assert num_states > 1
    _get_valid_seeds(seeds=seeds, num_states=num_states)

    states = [
        random_density_matrix(
            dims=2**num_qubits,
            rank=2,  # TODO rank
            method="Hilbert-Schmidt",  # TODO method
            seed=seeds[_],
        )
        for _ in range(num_states)
    ]
    logger = logging.getLogger(__name__)
    logger.info(
        f"{num_states} random {num_qubits}-qubit states (Density matrix) are generated"
    )
    return states


def gen_noisy_states(
    num_qubits: int,
    num_states: int,
    seeds: list[int] = [],
    noise_level: float = 0.01,
    noise_rank: int = 2,
    noise_seeds: list[int] = [],
    **kwargs,
):
    logger = logging.getLogger(__name__)
    assert num_qubits > 0
    assert num_states > 1
    # TODO Sparse state generation
    _get_valid_seeds(seeds=seeds, num_states=num_states)
    _get_valid_seeds(seeds=noise_seeds, num_states=num_states)

    # Initialize a set of pure states
    states = [
        random_statevector(
            2**num_qubits,
            seed=seeds[_],
        )
        for _ in range(num_states)
    ]

    # TODO Check orthogonality

    # Turn the pure states to density matrices
    dense_states = [DensityMatrix(states[i]) for i in range(num_states)]

    # Create a different set of random density matrices
    disturbance_states = [
        random_density_matrix(
            dims=2**num_qubits,
            rank=noise_rank,
            method="Hilbert-Schmidt",  # TODO method
            seed=seeds[_],
        )
        for _ in range(num_states)
    ]
    # Combine these

    combined_states = [
        (1 - noise_level) * dense_states[_].data
        + noise_level * disturbance_states[_].data
        for _ in range(num_states)
    ]

    combined_states = [DensityMatrix(combined_states[_]) for _ in range(num_states)]
    print(combined_states)

    logger.info(f"{num_states} random {num_qubits}-qubit noisy states are generated")
    return states, disturbance_states, combined_states


### Statevector
### Density matrix

## Noisy random states
### Density matrix
### Mix it

## Pure geometric states
### TODO What parameters

## Noisy geometric states?


# Property checking


## Check linear independence


def isLinearIndep(sv_list: list[Statevector]) -> bool:
    """Check the vectors are linearly independent."""
    l = []
    for i in range(len(sv_list)):
        l.append(sv_list[i].data)
    m = np.array(l)
    rank = np.linalg.matrix_rank(m)
    print("Rank =", rank)
    print("#states =", sv_list)
    return True


## Check trace

# Save matrices and parameters


def main():

    # TODO Unit tests
    # print(isLinearIndep())
    return


if __name__ == "__main__":
    main()
