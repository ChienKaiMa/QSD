using JuMP
using CPLEX

# Create a model
model = Model(CPLEX.Optimizer)

# Define the variables
@variable(model, x >= 0)
@variable(model, y >= 0)

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