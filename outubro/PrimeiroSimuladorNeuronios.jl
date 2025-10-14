using Plots, Random
Random.seed!(69)

const a = 0.95
const b = 0.1
const c = 5
const n = 20
const ganho = 4
const v0 = 30
const iteracoes = 5_000
const perda = 0.7

# https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
# plot(ϕ, xlim = (0,100), ylim = (0,1))
ϕ(v) = a ./ (1 .+ exp.(-b * v .+ c)) .+ 0.03

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
    W::Matrix{Int16} = fill(ganho, n, n)

    # Parâmetros da energia
    energias = fill(v0, n)
    dados = zeros(Int8, iteracoes, n)

    for i in 1:iteracoes
        # Ativa os neurônios com a regra probabilística
        probs = ϕ(energias)
        rands = rand(n)

        # Detecta os que foram ativadas e registra na matriz
        ativaram = findall(probs .> rands)
        pulsos = zeros(Int8, n)
        pulsos[ativaram] .= 1
        dados[i, :] = pulsos

        # Transfere energia para os adjacentes (segundo a matriz W)
        for i in ativaram
            for j in 1:n
                energias[j] += W[i, j]
            end
        end

        # Zera a energia do que ativou
        energias[ativaram] .= 0

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
        ), matriz
end

function simulaNeuroniosPerda()
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
    W::Matrix{Float16} = fill(Float16(ganho), n, n)

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

        # Transfere energia para os adjacentes (segundo a matriz W)
        for i in ativaram
            for j in 1:n
                energias[j] += W[i, j]
            end
        end

        # Zera a energia do que ativou
        energias[ativaram] .= 0
        energias = ρ(energias)

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
        ), matriz
end

p1, m1 = simulaNeuroniosPerda()
p2, m2 = simulaNeuronios()
