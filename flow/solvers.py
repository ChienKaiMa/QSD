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
        logger.info(f"{num_states} random states ({state_type}) are generated")
        return states
