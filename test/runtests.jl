using Test
using LinearAlgebra
using OptimalSmoothing

include(joinpath(@__DIR__, "..", "examples", "lorenz_model.jl"))

@testset "a linear map: the control of least norm in one iteration" begin

    # ---- x_{n+1} = M x_n + w_n f_n, so that the end is linear in the control ----
    M  = [1.01 0.02 0.0; -0.02 0.99 0.01; 0.0 0.01 0.97]
    N  = 40
    w  = [sin(π * n / (N + 1))^2 for n in 1:N]
    a  = [1.0, -2.0, 0.5]
    b  = [0.3, 0.1, -0.4]
    Φ  = f  -> foldl((x, n) -> M * x .+ w[n] .* f[:, n], 1:N; init=a)
    A  = δf -> foldl((x, n) -> M * x .+ w[n] .* δf[:, n], 1:N; init=zeros(3))
    Aᵀ = v  -> hcat([w[n] .* (M')^(N - n) * v for n in 1:N]...)

    # ---- the control of least norm, from the matrix of A ----
    Amat = hcat([A(reshape(e, 3, N)) for e in eachcol(Matrix(1.0I, 3N, 3N))]...)
    f★   = reshape(pinv(Amat) * (b .- Φ(zeros(3, N))), 3, N)

    sol = bridge(Φ, A, Aᵀ, b, zeros(3, N); tol=1e-10, cgtol=1e-13)
    @test norm(sol.f .- f★) < 1e-8 * norm(f★)
    @test length(sol.miss) <= 3
end

@testset "Lorenz: the transpose and the bridge" begin

    # ---- a state on the attractor, and a target off its continuation ----
    τ  = 0.25
    N  = 100
    a  = trajectory([1.0, 1.0, 1.0], 0.005, 4000)[:, end]
    b  = trajectory(a, 2τ / N, N)[:, end] .+ [0.8, -0.5, 0.3]

    Φ, A, Aᵀ, t, c, path = window_problem(a, τ, N)
    Φ(zeros(3, 2N + 1))

    # ---- the transpose, in the inner product of Simpson's rule ----
    δf = [sin(i + 3j) for i in 1:3, j in 1:2N + 1]
    v  = [0.3, -1.2, 0.7]
    @test dot(v, A(δf)) ≈ sum((Aᵀ(v) .* δf) .* c')

    # ---- the bridge lands on the target, with the control the transpose of the multiplier ----
    sol = bridge(Φ, A, Aᵀ, b, zeros(3, 2N + 1); tol=1e-9, cgtol=1e-10)
    @test norm(Φ(sol.f) .- b) < 1e-9 * norm(b)
    @test norm(sol.f .- Aᵀ(sol.ν)) < 1e-7 * norm(sol.f)
end
