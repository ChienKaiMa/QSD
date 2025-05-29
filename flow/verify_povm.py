import numpy as np
from scipy.linalg import ishermitian


def verify_povm(povm, rtol=1e-5):
    """Our POVM is a set of rank-1 vectors."""
    # TODO
    # Check shape
    # Check operator sum is identity
    ops = []
    for m in povm:
        # Add "None" to transpose
        # https://stackoverflow.com/a/11885718/13518808
        op = np.multiply(m[None].T.conj(), m)
        ops.append(op)
    # Check other properties
    return eq_opsum_id(ops, rtol=rtol)

def verify_povm_matrix(povm, rtol=1e-4):
    """Here the POVM is expressed as a list of Hermitian matrices"""
    # Check Hermitian
    for i in range(len(povm)):
        # May return False is you use rtol here.
        # I haven't figured out why.
        print(type(povm[i]))
        print("shape =", np.shape(povm[i]))
        print(povm[i])
        if not ishermitian(povm[i], atol=1e-10):
            print(f"{i} is not Hermitian")
            return False
    # Check completeness
    shape = povm[0].shape[0]
    # Omit small off-diagonal terms
    return np.allclose(np.diagonal(sum(povm)), np.diagonal(np.identity(shape)), rtol=rtol)


def eq_opsum_id(ops, rtol=1e-5):
    """Check if opsum equals to id
    rtol: rtol for numpy.allclose function. (default: 1e-5)
    The default value for rtol should follow numpy's default value.
    """
    opsum = 0
    print(type(ops[0]))
    shape = ops[0].shape[0]
    for op in ops:
        opsum += op
    # TODO potential bugfixes
    return np.allclose(opsum, np.identity(shape), rtol=rtol)
