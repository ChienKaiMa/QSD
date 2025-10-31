from problem_spec import *
import scipy

# TODO Create state preparation circuits from ProblemSpec

def binaryCode(a):

    return


def grayCode(a):
    # Add an extra wrapper in case we want to change the dependencies
    import graycode

    return graycode.tc_to_gray_code(a)
    # code = graycode.gen_gray_codes(3)
    # return code[a]


def mab(k, a, b):
    return 2 ** (-k) * (-1) ** (b & grayCode(a))


def phi_a(k, a, theta):
    l = [mab(k, a, b) * theta[b] for b in range(2**k)]
    return sum(l)


def prepPureEnsemble(probList: list[float], svList: list[Statevector]):
    """
    probList: List[], the a priori probabilities
    """

    return


def prepMixedEnsemble(mixList: list[DensityMatrix]):

    return


def uniformCtrlSingleRot(phi: float, theta: float):
    # circuit.ry(theta)
    # circuit.rz(phi)
    return


def uniformCtrlRot(numQubits: int, thetaList: list[float]):
    """Uniformly controlled rotation gate
    numQubits: int, = k-fold
    """
    phiList = [sum()]
    return


def prepDenseByMatrixElements(mat):
    # Density matrix state preparation in terms of matrix elements in Bo-Hung's paper
    # mat is probably a numpy array
    c = scipy.linalg.cholesky(mat, overwrite_a=False, lower=True)

    # Explore sparsity?
    # Remove all-zero columns
    # Reference:
    # https://www.geeksforgeeks.org/how-to-remove-array-rows-that-contain-only-0-using-numpy/
    c = c[~np.all(c == 0, axis=0)]
    # Method from IV.A
    # Mixture of pure state

    return