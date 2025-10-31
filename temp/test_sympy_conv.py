# Test Sympy + Convex optimization
from sympy import re, im, I, E, symbols

from sympy.solvers import (
    diophantine,
    solveset,
    solve_rational_inequalities,
    solve_linear_system_LU,
)
from sympy.solvers import linsolve, Complexes

from sympy import sqrt
from sympy.physics.quantum import Bra, Ket, qapply

superpos = (Ket("Dead") + Ket("Alive")) / sqrt(2)
d = qapply(Bra("Dead") * superpos)

print(d)
mySubs = {
    Bra("Dead") * Ket("Dead"): 1,
    Bra("Dead") * Ket("Alive"): 0,
}  ##plus other bindings
d.xreplace(mySubs)

print(d.xreplace(mySubs))

# print(Complexes)
