using Plots, Random
Random.seed!(69)

const a = 0.95
const b = 0.1
const c = 0
const n = 100
const ganho = 5
const v0 = 1
const iteracoes = 1_000
const perda = 0.8

# https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
# plot(ϕ, xlim = (0,100), ylim = (0,1))
ϕ(v) = a ./ (1 .+ exp.(-b * v .+ c))
# ϕ(v) = (v <= 5) ? v / 5 : 1
plot(ϕ, 0:1