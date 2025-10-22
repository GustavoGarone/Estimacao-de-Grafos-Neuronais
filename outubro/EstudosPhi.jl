using Plots, Random, StatsBase
Random.seed!(69)

const n = 10
const ganho = 5
const v0 = 1
const iteracoes = 100
const M = 10
const perda = 0.8

# https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
# plot(ϕ, xlim = (0,100), ylim = (0,1))

function simulaNeuronios(ϕ)
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

    vs::Vector{Vector{Float64}} = []


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
        append!(vs, [energias])

    end

    return reduce(hcat, vs)
end

function mediaEnergia(v)
    return 1 / (n * iteracoes) * sum(v)
end

function analisaPhi()
    """
    Analisa diferentes funções ϕ para a ativação dos neurônios.
    """

    maps = []
    for c in 0:10
        println(c)
        energias = zeros(100, 100) * NaN
        Threads.@threads for a in 0.01:0.01:1
            for b in 0.01:0.01:1
                ϕ(v) = a ./ (1 .+ exp.(-b * v .+ c))
                me = zeros(M)
                for i in 1:M
                    vs = simulaNeuronios(ϕ)
                    me[i] = mediaEnergia(vs)
                end
                energias[Int(round(a * 100)), Int(round(b * 100))] = ϕ(mean(me))
            end
        end
        push!(
            maps, heatmap(
                energias;
                color = :viridis,
                xlabel = "b",
                ylabel = "a",
                title = "Phi da média da energia dos neurônios para c = $c",
                legend = :right,
                clims = (0, 1)
            )
        )
    end
    return maps
end

plts = analisaPhi()

