
using JuMP
using CPLEX

# https://stackoverflow.com/questions/57270276/identity-matrix-in-julia
using LinearAlgebra

"""
Reproduce the example of Eldar's paper in 2003.
"""
function reproduce_2003_Eldar_results()
    print("shit\n")
    # np.set_printoptions(precision=3)
    phi1 = (1 / sqrt(3)) * Array{ComplexF64,1}([1.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im])
    phi2 = (1 / sqrt(2)) * Array{ComplexF64,1}([1.0 + 0.0im, 1.0 + 0.0im, 0.0 + 0.0im])
    phi3 = (1 / sqrt(2)) * Array{ComplexF64,1}([0.0 + 0.0im, 1.0 + 0.0im, 1.0 + 0.0im])

    # https://stackoverflow.com/questions/38819425/how-to-specify-the-format-for-printing-an-array-of-floats-in-julia
    # @show(round.(phi1; digits=3))

    # https://stackoverflow.com/questions/39586830/concatenating-arrays-in-julia
    psi = hcat(phi1, phi2, phi3)
    # round.(psi; digits=3)
    recip_psi = LinearAlgebra.pinv(psi)
    # print(round.(recip_psi; digits=3), "\n")

    # https://stackoverflow.com/questions/37482968/selecting-columns-rows-of-a-matrix-in-julia
    # https://stackoverflow.com/questions/44591481/julia-outer-product-function
    num_states = 3
    for i in 1:num_states
        # print(round.(recip_psi[i, :]; digits=3), "\n")

        print(round.(recip_psi[i, :] .* recip_psi[i, :]'; digits=3), "\n")
    end

    # prior_prob = np.multiply(-1 / 3, [1, 1, 1])
    # @variables
end

"""

    n = 3
    # Measurement operators
    q = []
    for i in range(n):
        q.append(np.outer(recip_psi[i], recip_psi[i]).round(1))
    q = np.array(q)

    I = np.identity(n)
    p = cp.Variable(n)
    objective = cp.Minimize(1 + cp.sum(prior_prob @ p))
    constraints = [
        0 <= p[0],
        0 <= p[1],
        0 <= p[2],
        I - p[0] * q[0] - p[1] * q[1] - p[2] * q[2] >> 0,  # Matrix inequality uses >>
    ]

    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print("Result =", result.round(3))
    # An acceptable optimal solution
    sol = p.value.round(4)
    print("Solution =", sol)
    pi1 = I - sol[0] * q[0] - sol[1] * q[1] - sol[2] * q[2]  # Positive semidefinite
    print(pi1.round(5))
    # Wrong answer if we over postprocess the solution
    sol_overround = p.value.round(2)
    print("Overprocessed solution =", sol_overround)
    pi1_overround = (
        I - sol_overround[0] * q[0] - sol_overround[1] * q[1] - sol_overround[2] * q[2]
        )  # Not positive semidefinite
        print(pi1_overround.round(5))
        
        # The optimal Lagrange multiplier for a constraint
        # is stored in constraint.dual_value.
    # print(constraints[0].dual_value)









# Create a model
model = Model(CPLEX.Optimizer)

# Define the variables
@variable(model, x >= 0)
@variable(model, y >= 0)

num_qubits = 3
E_i = Array{ComplexF64, 2 ^ num_qubits}
zero
# Add constraint
@constraint(model, x + y <= 1)

# Set objective
@objective(model, Max, x + 2y)

# Optimize
optimize!(model)

# Print results
println("Optimal solution:")
println("x = ", value(x))
println("y = ", value(y))
"""
function placeholder()
    print(0)
end

reproduce_2003_Eldar_results()