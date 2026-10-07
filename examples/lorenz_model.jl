# The Lorenz system over a window -τ ≤ t ≤ τ, forced by w f, with w = (1 - t²/τ²)³, integrated by the
# classical Runge-Kutta scheme of order four. Builds the end map Φ, its derivative A and its
# transpose Aᵀ for OptimalSmoothing.bridge. The control is sampled at the steps and at the half
# steps, 2N + 1 samples for N steps, in a 3 × (2N + 1) matrix. Its inner product is that of Simpson's
# rule, and A and Aᵀ are the exact derivative of the discrete map and its transpose in it.

using LinearAlgebra

const σ = 10.0
const ρ = 28.0
const β = 8 / 3

lorenz(x)   = [σ * (x[2] - x[1]), x[1] * (ρ - x[3]) - x[2], x[1] * x[2] - β * x[3]]
jacobian(x) = [-σ σ 0.0; ρ - x[3] -1.0 -x[1]; x[2] x[1] -β]

window(t, τ) = abs(t) < τ ? (1 - (t / τ)^2)^3 : 0.0

# one step from x, with the forcing g0, gh and g1 at the start, the middle and the end of the step;
# returns the new state and the four points where the stages are evaluated
function rk4(x, h, g0, gh, g1)
    k1 = lorenz(x) .+ g0
    y2 = x .+ (h / 2) .* k1
    k2 = lorenz(y2) .+ gh
    y3 = x .+ (h / 2) .* k2
    k3 = lorenz(y3) .+ gh
    y4 = x .+ h .* k3
    k4 = lorenz(y4) .+ g1
    return x .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4), (x, y2, y3, y4)
end

# the trajectory from x over n steps of h, without forcing, as a 3 × (n + 1) matrix
function trajectory(x, h, n)
    X = zeros(3, n + 1)
    X[:, 1] = x
    for i in 1:n
        X[:, i + 1] = rk4(X[:, i], h, 0.0, 0.0, 0.0)[1]
    end
    return X
end

"""
    window_problem(a, τ, N; shape=window) -> (Φ, A, Aᵀ, t, c, path)

The functions `Φ`, `A` and `Aᵀ` of `bridge` for the window -τ ≤ t ≤ τ from the state `a`, in `N`
steps, with the weight `shape(t, τ)` of the forcing; `t` the times of the samples of the control, `c`
the weights of their inner product, and `path()` the states at the steps of the last call to `Φ`, a
3 × (N + 1) matrix.
"""
function window_problem(a, τ, N; shape=window)
    h      = 2τ / N
    t      = range(-τ, τ; length=2N + 1)
    w      = shape.(t, τ)
    c      = [j == 1 || j == 2N + 1 ? h / 6 : iseven(j) ? 2h / 3 : h / 3 for j in 1:2N + 1]
    stages = Vector{NTuple{4, Vector{Float64}}}(undef, N)
    last   = copy(a)

    # ---- the end of the state with the control f, storing the points of the stages ----
    function Φ(f)
        x = copy(a)
        for n in 1:N
            x, stages[n] = rk4(x, h, w[2n - 1] .* f[:, 2n - 1], w[2n] .* f[:, 2n], w[2n + 1] .* f[:, 2n + 1])
        end
        last .= x
        return x
    end

    # ---- the change of the end due to δf, by the tangent of the scheme ----
    function A(δf)
        δx = zeros(3)
        for n in 1:N
            J  = jacobian.(stages[n])
            k1 = J[1] * δx .+ w[2n - 1] .* δf[:, 2n - 1]
            k2 = J[2] * (δx .+ (h / 2) .* k1) .+ w[2n] .* δf[:, 2n]
            k3 = J[3] * (δx .+ (h / 2) .* k2) .+ w[2n] .* δf[:, 2n]
            k4 = J[4] * (δx .+ h .* k3) .+ w[2n + 1] .* δf[:, 2n + 1]
            δx = δx .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        end
        return δx
    end

    # ---- the transpose of A in the inner product of Simpson's rule, backward over the steps ----
    function Aᵀ(v)
        g = zeros(3, 2N + 1)
        λ = copy(v)
        for n in N:-1:1
            J  = jacobian.(stages[n])
            μ4 = (h / 6) .* λ
            a4 = J[4]' * μ4
            μ3 = (h / 3) .* λ .+ h .* a4
            a3 = J[3]' * μ3
            μ2 = (h / 3) .* λ .+ (h / 2) .* a3
            a2 = J[2]' * μ2
            μ1 = (h / 6) .* λ .+ (h / 2) .* a2
            a1 = J[1]' * μ1

            # the forcing at the start, the middle and the end of the step
            g[:, 2n - 1] .+= μ1
            g[:, 2n]     .+= μ2 .+ μ3
            g[:, 2n + 1] .+= μ4
            λ = λ .+ a1 .+ a2 .+ a3 .+ a4
        end
        return g .* (w ./ c)'
    end

    # ---- the states at the steps of the last call to Φ ----
    path() = hcat([stages[n][1] for n in 1:N]..., last)

    return Φ, A, Aᵀ, t, c, path
end
