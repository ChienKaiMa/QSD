from solvers import ProblemSpec

import rlcompleter
import readline
readline.parse_and_bind("tab: complete")

import sys
import cplex
import docplex


def blah(filename):
    c = cplex.Cplex(filename)

    # TODO Change output stream
    c.set_log_stream(sys.stdout)
    c.set_results_stream(sys.stdout)

    solve_instance = c.register_calback

    c.objective.set_sense(c.objective.sense.minimize)
    c.quadratic_constraints.add(
        lin_expr="",
        quad_expr="",
        sense,
        rhs,
        name,
    )

    indices = c.variables.add(names=[str(i) for i in range(3)])
    c.objective.set_quadratic(
        [cplex.SparsePair(ind=[0, 1, 2], val=[1.0, -2.0, 0.5]),
         cplex.SparsePair(ind=[0, 1], val=[-2.0, -1.0]),
         cplex.SparsePair(ind=[0, 2], val=[0.5, -3.0])]
    )
    for q in c.objective.get_quadratic():
        print(q)
    
    c.objective.set_quadratic([1.0, 2.0, 3.0])
    for q in c.objective.get_quadratic():
        print(q)
    c.objective.get_indices(
    c.objective.get_linear(
    c.objective.get_name()
    c.objective.get_num_quadratic_nonzeros()
    c.objective.get_num_quadratic_variables()
    c.objective.get_offset()
    c.objective.get_quadratic(
    c.objective.get_quadratic_coefficients(
    c.objective.get_sense()
    c.objective.sense
    c.objective.set_linear(
    c.objective.set_name(
    c.objective.set_offset(
    c.objective.set_quadratic(
    c.objective.set_quadratic_coefficients(
    c.objective.set_sense(
    ## linear_constraints
    ## objective
    ## cplex.Cplex.solution()
    ## cplex.Cplex.set_problem_name()
    ## cplex.Cplex.set_problem_type()
    ## cplex.Cplex.read()
    ## cplex.Cplex.solve()
    ## cplex.Cplex.cleanup()
