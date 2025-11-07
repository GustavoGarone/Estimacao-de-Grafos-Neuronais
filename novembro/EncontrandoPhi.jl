# Neste arquivo, buscamos encontrar funções de ativação ϕ com comportamento
# interessante. Definimos como linear, e buscamos padrões entre a relação do
# coeficiente angular e
using Plots, Random, StatsBase, LinearAlgebra
const a = 1
const ϵ = 0.05
const β = 0.7
const c = log(a / ϵ - 1)
const n = 10
const exc = 1
const pexc = 0.2 # Valores exemplo do diogo
const inib = -1
const pinib = 0.05
const v0 = 1
const iteracoes = 1000
const M = 10
const perda = 1

# https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
# plot(ϕ, xlim = (0,100), ylim = (0,1))

function criaGrafo()
    W::Matrix{Float32} = fill(Float32(0), n, n)
    for i in 1:n
        for j in 1:n
            if i != j
                r = rand()
                if r < pexc
                    W[i, j] = exc
                elseif r < pexc + pinib
                    W[i, j] = inib
                end
            end
        end
    end
    return W
end

function calculaM(W)
    # return sum(W) / (n * (n - 1))
    # Versão Diogo
    return max(1, sum(abs.(W)) / n)
end

W = criaGrafo()
m = calculaM(W)
b = (c - log(a / β - 1)) / m

#https://www.desmos.com/calculator/v2ebrbvwg8

function ϕ(v)
    return a / (1 + exp(-b * v + c))
end

function simulaNeuroniosDiagnostico(seed)
    """
    Nesse modelo, os neurônios, quando ativam, enviam cargas para os neurônios
    conectoados a ele no grafo e, ao final da iteraão, todos que foram ativados são
    zerados.
    Alternativa, pode-se modelar um sistema em que a matriz W tem diagonal nula e,
    quando um neurônio se ativa, na iteração de disperção das energias,
    não ganha energia. Todavia, zeramos todos os neurônios assim que
    são ativados e só em seguida dispersamos a energia para os demais. Nesse novo
    modelo, a cada interação, um neurônio que se ativou pode finalizar a interação
    com energia maior do que 0.
    """
    Random.seed!(seed)

    # Seja W a matriz que representa o grafo ponderado orientado das interações
    # Usamos um grafo completo

    # plotphi = plot(
    #     ϕ, xlims = (-2, 12), ylims = (-0.1, 1.1),
    #     title = "Função de ativação ϕ", xlabel = "Energia v",
    #     ylabel = "Probabilidade de ativação ϕ(v)",
    #     framestyle = :zerolines
    # )
    # display(plotphi)

    # Função de perda
    ρ(v) = v * perda


    # Parâmetros da energia
    energias = fill(v0, n)
    dados = zeros(iteracoes, n)

    # Testes
    uniformes = []
    vs::Vector{Vector{Float64}} = []


    for i in 1:iteracoes
        # Ativa os neurônios com a regra probabilística
        probs = ϕ.(energias)
        rands = rand(n)
        append!(uniformes, rands)

        # Detecta os que foram ativadas e registra na matriz
        ativaram = findall(probs .> rands)
        pulsos = zeros(n)
        pulsos[ativaram] .= 1
        dados[i, :] = pulsos

        # Ocasiona perda
        energias = ρ(energias)

        # Transfere energia para os adjacentes (segundo a matriz W)
        for i in ativaram
            energias += W[i, :]
        end


        # Zera a energia do que ativou
        energias[ativaram] .= 0
        append!(vs, [energias])

    end

    matriz = permutedims(dados)
    # Columnwise mean of vs
    # mean_v = mean(reduce(hcat, vs), dims = 2)

    vvs = reduce(hcat, vs)
    mapa = heatmap(
        matriz;
        yflip = true,
        color = :binary,
        xlabel = "Instante",
        ylabel = "Neurônio",
        title = "Instantes de ativações dos neurônios",
        legend = false
    )
    enerplot = plot(
        vec(mean(vvs, dims = 1));
        label = "Média",
        xlabel = "Instante",
        ylabel = "Energia",

    )
    plot!(
        enerplot,
        vec(maximum(vvs, dims = 1));
        label = "Máxima",
    )
    plot!(
        enerplot,
        vec(minimum(vvs, dims = 1));
        label = "Mínima",
    )
    # Plota as ativações dos neurônios
    return mapa, matriz, uniformes, vvs, m, W, enerplot
end

p1, m1, unif, v1, m, w, p2 = simulaNeuroniosDiagnostico(42)
pphi = plot(
    ϕ, xlims = (-2, 12), ylims = (-0.1, 1.1),
    title = "Função de ativação ϕ", xlabel = "Energia v",
    ylabel = "Probabilidade de ativação ϕ(v)",
    label = "",
    framestyle = :zerolines,
)
hline!([β], label = "β = $β", linestyle = :dash)
vline!([m], label = "m = $m", linestyle = :dash)
