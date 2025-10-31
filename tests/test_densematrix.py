from problem_spec import *


if __name__ == "__main__":
    ps = ProblemSpec(
        num_qubits=3,
        num_states=4,
        seed=5,
        case_id=''
    )

    a, b, c = ProblemSpec.gen_noisy_states(
        num_qubits=2,
        num_states=4,
        noise_level=0.01,
        noise_rank=2,
    )
    # print(a)
    # print()
    # print(b)
    # print()
    # print(c)
    # print()
    pass
