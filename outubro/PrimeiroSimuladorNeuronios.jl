using Plots, Random, StatsBase
Random.seed!(69)

const a = 1
const b = 1
const c = 0
const n = 10
const ganho = 1
const v0 = 1
const iteracoes = 100
const perda = 0.8

# https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
# plot(ϕ, xlim = (0,100), ylim = (0,1))

# Phi sinoidal
ϕ(v) = a ./ (1 .+ exp.(-b * v .+ c))

# Phi linear
# ϕ(v) = (v <= 10) ? v / 10 : 1

# ϕ(v) = min(1, 2v / 7)


# plot(0:10, ϕ)

function simulaNeuronios()
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

    # Seja W a matriz que representa o grafo ponderado orientado das interações
    # Usamos um grafo completo
    W::Matrix{Float32} = fill(Float32(ganho), n, n)

    # Função de perda
    ρ(v) = v * perda

    # https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
    # plot(ϕ, xlim = (0,100), ylim = (0,1))

    # Parâmetros da energia
    energias = fill(v0, n)
    dados = zeros(iteracoes, n)

    for i in 1:iteracoes
        # Ativa os neurônios com a regra probabilística
        probs = ϕ.(energias)
        rands = rand(n)

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

    end

    matriz = permutedims(dados)

    return matriz
end

function simulaNeuroniosDiagnostico()
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

    # Seja W a matriz que representa o grafo ponderado orientado das interações
    # Usamos um grafo completo
    W::Matrix{Float32} = fill(Float32(ganho), n, n)

    # Função de perda
    ρ(v) = v * perda

    # https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
    # plot(ϕ, xlim = (0,100), ylim = (0,1))

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

    # Plota as ativações dos neurônios
    return heatmap(
            matriz;
            yflip = true,
            color = :binary,
            xlabel = "Instante",
            ylabel = "Neurônio",
            title = "Instantes de ativações dos neurônios",
            legend = false
        ), matriz, uniformes, reduce(hcat, vs)
end

# p1, m1, u1, v1 = simulaNeuroniosDiagnostico()
