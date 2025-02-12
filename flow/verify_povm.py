import numpy as np


def verify_povm(povm):
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
    return eq_opsum_id(ops)


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
