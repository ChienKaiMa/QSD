# - Necessary - #
using Alpine
using JuMP
using Gurobi
using Ipopt
using Juniper

# - Additional - #
# using CPLEX
# using HiGHS
# using Pavito

# include("JuMP_models.jl")
include("optimizers.jl")

# Choose underlying solvers for Alpine
nlp_solver = get_ipopt() # local continuous solver
mip_solver = get_gurobi() # convex mip solver
minlp_solver = get_juniper(mip_solver, nlp_solver) # local mixed-intger solver

#= Global solver
 Hints: 
 => Try different integer values (>=4) for `partition_scaling_factor` to potentially observe 
    better Alpine run times (can be instance specific).
 => Choose `presolve_bt` to `false` if you prefer the bound tightening (OBBT) presolve to be turned off. 
 => If you prefer to use Alpine for only OBBT presolve, without any paritioning applied to the 
    nonlinear terms, include option "apply_partitioning" below and set it to false. 
=#

const alpine = JuMP.optimizer_with_attributes(
    Alpine.Optimizer,
    # "minlp_solver" => minlp_solver,
    "nlp_solver" => nlp_solver,
    "mip_solver" => mip_solver,
    "presolve_bt" => true,
    "apply_partitioning" => true,
    "partition_scaling_factor" => 10,
)

m = nlp3(solver = alpine)

JuMP.optimize!(m)
# Alpine.variable_values(m)






using JuMP, Alpine, Ipopt, HiGHS
import MathOptInterface as MOI
import LinearAlgebra
ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
highs = optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
# model = Model(
#     optimizer_with_attributes(
#         Alpine.Optimizer,
#         "nlp_solver" => ipopt,
#         "mip_solver" => highs,
#     ),
# )
m = JuMP.Model(solver)

@variable(m, 0 <= objvar <= 5)
@variable(m, 1 <= x[1:2] <= 5)

@NLconstraint(
    m,
    e1,
    (2.545724188 - x[1])^2 + (9.983058643 - x[2])^2 - (objvar)^2 >= 0.0
)

# MOI.dimension(MOI.Reals(4))
# model = MOI.Nonlinear.Model()
# model.dimension(ComplexF32)
# MOI.Nonlinear.set_objective(model)
# x = MOI.HermitianPositiveSemidefiniteConeTriangle(4)
# MOI.dimension(MOI.Complex(4))
# x = MOI.PositiveSemidefiniteConeSquare(4)
# MOI.NumberOfVariables()
# MOI.add_variables(model, 4)



# q.append(
    #     [
    #         [1, -1, v[0]],
    #         [-1, 1, -v[0]],
    #         [v[0], -v[0], v[0] ** 2],
    #     ]
    # )
# 
    # q.append(
    #     [
    #         [0, 0, 0],
    #         [0, 2, -1.414 * v[1]],
    #         [0, -1.414 * v[1], v[1] ** 2],
    #     ]
    # )
    q.append(
        [
            [p[0], -p[0], p[0] * v[0]],
            [-p[0], p[0], -p[0] * v[0]],
            [p[0] * v[0], -p[0] * v[0], p[0] * v[0] ** 2],
        ]
    )

    q.append(
        [
            [0, 0, 0],
            [0, 2, -1.414 * v[1]],
            [0, -1.414 * v[1], v[1] ** 2],
        ]
    )

    
    # cp.outer([1.0, -1, v[0]], [1.0, -1, v[0]])
    # q[1] = cp.outer([0.0, 1.414, v[1]], [0.0, 1.414, v[1]])