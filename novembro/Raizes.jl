using Roots
using Printf

# Função f(b, u)
f(b, u) = (b^2 * exp(log(19) - b*u)) / ( (exp(log(19) - b*u) + 1)^2 ) - 0.7

# Tenta achar soluções reais para b para determinado u
function acha_raiz(u)
    bs = []
    for (x1, x2) in zip(-50:0.5:50, -49.5:0.5:50.5)
        if f(x1,u) * f(x2,u) < 0
            try
                root = find_zero(b -> f(b,u), (x1,x2), Bisection(), verbose=false)
                push!(bs, root)
            catch
            end
        end
    end
    return sort(unique(bs))
end

for i in 0:0.1:1
    raizes = acha_raiz(i)
    raizes = round.(raizes, digits=2)
    println("|  $i  |  $(raizes[1]) | $(raizes[2]) | ")
end