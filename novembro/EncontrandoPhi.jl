# Neste arquivo, buscamos encontrar funções de ativação ϕ com comportamento
# interessante. Definimos como linear, e buscamos padrões entre a relação do
# coeficiente angular e
using Plots, Random, StatsBase, LinearAlgebra, LaTeXStrings

function analisaPhi(semente::Int, n::Int, iteracoes::Int)
    Random.seed!(semente)
    # Parâmetros
    a = 1.0 # Probabilidade máxima de disparo
    ϵ = 0.0 # Probbabilidade de disparo espontâneo
    α = 0.0 # Probabilidade mínima de disparo
    β = 0.7 # Probabilidade de disparo média desejada
    c = log(a / ϵ - 1)
    exc = 1
    pexc = 0.2 # Valores exemplo do diogo
    inib = -1
    pinib = 0.05
    v0 = 5
    M = 10
    perda = 0.96

    function criaGrafo()
        W = fill(Float32(0), n, n)
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

    W::Matrix{Float32} = criaGrafo()
    m = calculaM(W)
    b = (c - log(a / β - 1)) / m

    if ϵ < α
        error("Erro: ϵ deve ser maior ou igual a α")
    end
    function ϕ(v)
        return max(a / (1 + exp(-b * v + c)), α)
    end

    function simulaNeuroniosDiagnostico()
        """
        Nesse modelo, os neurônios, quando ativam, enviam cargas para os neurônios
        conectados a ele no grafo e, ao final da iteraão, todos que foram ativados são
        zerados.
        Alternativa, pode-se modelar um sistema em que a matriz W tem diagonal nula e,
        quando um neurônio se ativa, na iteração de disperção das energias,
        não ganha energia. Todavia, zeramos todos os neurônios assim que
        são ativados e só em seguida dispersamos a energia para os demais. Nesse novo
        modelo, a cada interação, um neurônio que se ativou pode finalizar a interação
        com energia maior do que 0.
        """
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

        vvs = reduce(hcat, vs)
        meds = vec(mean(vvs, dims = 1)); mmeds = mean(meds)
        maxs = vec(maximum(vvs, dims = 1)); mmaxs = mean(maxs)
        mins = vec(minimum(vvs, dims = 1)); mmins = mean(mins)

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
            meds,
            label = "Média",
            xlabel = "Instante",
            ylabel = "Energia",
            legend = :outerbottom,
        )
        plot!(
            enerplot,
            maxs,
            label = "Máximo",
        )
        plot!(
            enerplot,
            mins,
            label = "Mínimo",
        )
        hline!(
            [mmeds],
            label = "Média Geral = $(round(mmeds, digits = 4))",
            linestyle = :dash
        )
        hline!(
            [mmaxs],
            label = "Média Máximos = $(round(mmaxs, digits = 4))",
            linestyle = :dash
        )
        hline!(
            [mmins],
            label = "Média Mínimos = $(round(mmins, digits = 4))",
            linestyle = :dash
        )

        pphi = plot(
            ϕ, xlims = (-3, 20), ylims = (-0.1, 1.1),
            title = "Função de ativação " * L"\varphi", xlabel = "Energia v",
            ylabel = "Probabilidade de ativação " * L"\varphi(v)",
            label = "",
            framestyle = :zerolines,
            legend = :bottomright
        )
        hline!([β], label = L"β = %$β", linestyle = :dash)
        vline!([m], label = L"m = %$m", linestyle = :dash)
        scatter!(
            [mmeds], [ϕ(mmeds)],
            label = L"\varphi(\bar{v}) = %$(round(ϕ(mmeds), digits=4))", color = :red
        )
        scatter!(
            [mmaxs], [ϕ(mmaxs)],
            label = L"\varphi(v_{max}) = %$(round(ϕ(mmaxs), digits=4))", color = :green
        )
        scatter!(
            [mmins], [ϕ(mmins)],
            label = L"\varphi(v_{min}) = %$(round(ϕ(mmins), digits=4))", color = :blue
        )

        # Plota as ativações dos neurônios
        return mapa, enerplot, pphi, matriz, uniformes, vvs, m, W
    end

    return simulaNeuroniosDiagnostico()
end

function analisaPhiMorte(semente::Int, n::Int, iteracoes::Int)
    Random.seed!(semente)
    # Parâmetros
    a = 1.0 # Probabilidade máxima de disparo
    ϵ = 0.0 # Probbabilidade de disparo espontâneo
    α = 0.0 # Probabilidade mínima de disparo
    β = 0.7 # Probabilidade de disparo média desejada
    c = log(a / ϵ - 1)
    exc = 1
    pexc = 0.2 # Valores exemplo do diogo
    inib = -1
    pinib = 0.05
    v0 = 5
    M = 10
    perda = 0.96

    function criaGrafo()
        W = fill(Float32(0), n, n)
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

    W::Matrix{Float32} = criaGrafo()
    m = calculaM(W)
    b = (c - log(a / β - 1)) / m

    if ϵ < α
        error("Erro: ϵ deve ser maior ou igual a α")
    end
    ϕ = nothing

    k = 0.01
    if ϵ <= 0
        c = log(a / k - 1)
        b = (c - log(a / β - 1)) / m
        println("b = $b, c = $c")
        ϕ = function (v)
            return max((a + k) * 1 / (1 + exp(-b * v + c)) - k * (1 + k), α)
        end
    else
        ϕ = function (v)
            return max(a / (1 + exp(-b * v + c)), α)
        end
    end

    function simulaNeuronios()
        # Função de perda
        ρ(v) = v * perda

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

        return matriz = permutedims(dados)
    end

    function simulaNeuroniosDiagnostico()
        """
        Nesse modelo, os neurônios, quando ativam, enviam cargas para os neurônios
        conectados a ele no grafo e, ao final da iteraão, todos que foram ativados são
        zerados.
        Alternativa, pode-se modelar um sistema em que a matriz W tem diagonal nula e,
        quando um neurônio se ativa, na iteração de disperção das energias,
        não ganha energia. Todavia, zeramos todos os neurônios assim que
        são ativados e só em seguida dispersamos a energia para os demais. Nesse novo
        modelo, a cada interação, um neurônio que se ativou pode finalizar a interação
        com energia maior do que 0.
        """
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

        vvs = reduce(hcat, vs)
        meds = vec(mean(vvs, dims = 1)); mmeds = mean(meds)
        maxs = vec(maximum(vvs, dims = 1)); mmaxs = mean(maxs)
        mins = vec(minimum(vvs, dims = 1)); mmins = mean(mins)

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
            meds,
            label = "Média",
            xlabel = "Instante",
            ylabel = "Energia",
            legend = :outerbottom,
        )
        plot!(
            enerplot,
            maxs,
            label = "Máximo",
        )
        plot!(
            enerplot,
            mins,
            label = "Mínimo",
        )
        hline!(
            [mmeds],
            label = "Média Geral = $(round(mmeds, digits = 4))",
            linestyle = :dash
        )
        hline!(
            [mmaxs],
            label = "Média Máximos = $(round(mmaxs, digits = 4))",
            linestyle = :dash
        )
        hline!(
            [mmins],
            label = "Média Mínimos = $(round(mmins, digits = 4))",
            linestyle = :dash
        )

        pphi = plot(
            ϕ, xlims = (-3, 20), ylims = (-0.1, 1.1),
            title = "Função de ativação " * L"\varphi", xlabel = "Energia v",
            ylabel = "Probabilidade de ativação " * L"\varphi(v)",
            label = "",
            framestyle = :zerolines,
            legend = :bottomright
        )
        hline!([β], label = L"β = %$β", linestyle = :dash)
        vline!([m], label = L"m = %$m", linestyle = :dash)
        scatter!(
            [mmeds], [ϕ(mmeds)],
            label = L"\varphi(\bar{v}) = %$(round(ϕ(mmeds), digits=4))", color = :red
        )
        scatter!(
            [mmaxs], [ϕ(mmaxs)],
            label = L"\varphi(v_{max}) = %$(round(ϕ(mmaxs), digits=4))", color = :green
        )
        scatter!(
            [mmins], [ϕ(mmins)],
            label = L"\varphi(v_{min}) = %$(round(ϕ(mmins), digits=4))", color = :blue
        )


        # Plota as ativações dos neurônios
        return mapa, enerplot, pphi, matriz, uniformes, vvs, m, W
    end
    return simulaNeuroniosDiagnostico()
end

analise = analisaPhi(10, 50, 1_000)
analiseMorte = analisaPhiMorte(10, 50, 1_000)
