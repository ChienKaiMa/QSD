import numpy as np


def vectors_to_povm(povm_vectors: list) -> np.array:
    """Expand rank-1 vectors to measurement operators"""
    # TODO assert input integrity
    povm = []
    for m in povm_vectors:
        povm.append(np.multiply(m[None].T.conj(), m))
    return np.array(povm)


def compute_event_probabilities(prior_prob, povm, state: np.array):
    """
    Compute the probabilities of all possible measured values based
    on a given quantum state and a given POVM.
    prior_prob: Prior probability of preparing the state, float
    povm: List of measurement operators
    state: Density matrix

    p * Tr(rho * M) for each M in povm
    Note that it does consider prior probabilities.
    """
    probs = []
    for m in povm:
        trace_value = np.trace(np.matmul(state, m)).item()
        # print(trace_value, flush=True)
        # TODO Find the potential bug
        # assert trace_value.real >= -1e-2
        assert abs(trace_value.imag) <= 1e-7
        probs.append(prior_prob * abs(trace_value.real))
    return probs

## def calculate_prob_matrix(
##     prior_probs, povm, states, bitstring_to_target_state, strings_used
## ):
##     probability_matrix = []
##     for i in range(len(states)):
##         probs = compute_event_probabilities(prior_probs[i], povm, states[i])
##         updated_probs = [0] * (len(prior_probs) + 1)
##         print("len(povm)", len(povm))
##         print("len(probs)", len(probs))
##         print("strings_used", strings_used)
##         for j in range(strings_used):
##             target_state_index = bitstring_to_target_state[j]
##             updated_probs[target_state_index] += probs[j]
##         # TODO
##         # Postprocessing
##         probability_matrix.append(updated_probs)
##     return probability_matrix
##
##
def calculate_prob_matrix_simple(prior_probs, povm, states):
    """Calculate the probabilities without mapping"""
    assert len(povm) == len(states) + 1
    probability_matrix = []
    for i in range(len(states)):
        probs = compute_event_probabilities(prior_probs[i], povm, states[i])
        ## updated_probs = [0] * (len(prior_probs) + 1)
        ## print("len(povm)", len(povm))
        ## print("len(probs)", len(probs))
        ## print("strings_used", strings_used)
        ## for j in range(len(povm)):
        ##     target_state_index = bitstring_to_target_state[j]
        ##     updated_probs[target_state_index] += probs[j]
        # TODO
        # Postprocessing
        probability_matrix.append(probs)
    return probability_matrix


def calculate_errors(prob_matrix):
    """Calculate the hyperparameters from the probability matrix."""
    prob_matrix = np.array(prob_matrix)
    shape = np.shape(prob_matrix)
    assert shape[0] + 1 == shape[1]
    assert len(prob_matrix[0][:-1]) == shape[0]

    alpha, beta = [], []
    for i in range(shape[0]):
        alpha.append(1 - (prob_matrix[i][i] / sum(prob_matrix[i, :-1])))

    for i in range(shape[0]):
        beta.append(1 - prob_matrix[i][i] / sum(prob_matrix[:, i]))

    return alpha, beta


if __name__ == "__main__":
    prior_prob = [1 / 3, 1 / 3, 1 / 3]
    povm_vectors = [np.diagonal(np.identity(3))]
    povm = vectors_to_povm(povm_vectors)
    print(povm)
    state = [
        [0.4, 0.2, 0.25],
        [0.2, 0.35, 0.3],
        [0.25, 0.3, 0.25],
    ]
    prob = compute_event_probabilities(prior_prob[0], povm, state)
    print(prob)
