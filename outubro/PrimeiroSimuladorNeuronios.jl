using Plots

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
    n = 20
    W::Matrix{Int16} = fill(Int16(4), n, n)

    # Definimos uma função para a probabilidade de disparar
    a = 0.95
    b = 0.1
    c = 5

    ϕ(v) = a / (1 + exp(-b * v + c))

    # https://www.desmos.com/calculator/qj06kcul6d?lang=pt-BR
    # plot(ϕ, xlim = (0,100), ylim = (0,1))

    # Parâmetros da energia
    v0 = 50

    energias = fill(v0, n)

    iteracoes = 100
    dados = zeros(Int8, iteracoes, n)

    for i in 1:iteracoes
        # Ativa os neurônios com a regra probabilística
        probs = ϕ.(energias)
        rands = rand(n)

        # Detecta os que foram ativadas e registra na matriz
        ativaram = findall(probs .> rands)
        pulsos = zeros(Int8, n)
        pulsos[ativaram] .= 1
        println(pulsos)
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

    matriz = reduce(vcat, permutedims.(dados))

    # Plota as ativações dos neurônios
    return heatmap(
            matriz';
            yflip = true,
            color = :binary,
            xlabel = "Instante",
            ylabel = "Neurônio",
            title = "Instantes de ativações dos neurônios",
            legend = false
        ), matriz
end

simulaNeuronios()
