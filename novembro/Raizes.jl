using Roots, PrettyTables, CommonSolve

raizes = []
for u in 0:0.1:1
    f(b) = (b^2 * exp(log(19) - b * u)) / ((exp(log(19) - b * u) + 1)^2)
    push!(raizes, find_zero(f, -10)
end

