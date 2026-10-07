# A bridge across the junction of a recurrence of the Lorenz system. A segment of trajectory that
# visits both wings of the attractor and returns close to its start is a guess for a periodic orbit,
# but it jumps where its end meets its start. The bridge replaces the segment near the junction by
# the trajectory with the least forcing that leaves the end and lands exactly on the start. The
# script compares the residual of the guess before and after, by the spectral derivative over the
# period, projects the forcing on the vector field and on the covariant Lyapunov vectors, shows how
# Newton's method and its conjugate gradients converge, and compares windows of different widths.
#
#     julia --project=examples examples/lorenz.jl
#
# Writes in examples/ the figures lorenz_recurrence.png, lorenz_jump.png, lorenz_guess.png,
# lorenz_residual.png, lorenz_forcing.png, lorenz_clv.png, lorenz_newton.png and lorenz_cg.png, for
# the window of half-width TAU, and lorenz_tau_guess.png, lorenz_tau_residual.png,
# lorenz_tau_forcing.png, lorenz_tau_clv.png and lorenz_tau_summary.png, for the windows TAUS, and
# lorenz_weight_shapes.png, lorenz_weight_residual.png, lorenz_weight_forcing.png and
# lorenz_weight_summary.png, for windows of different shapes.

using Printf
using LinearAlgebra
using CairoMakie
using OptimalSmoothing

include("lorenz_model.jl")


# ============================================================================ #
# Parameters                                                                   #
# ============================================================================ #

const H     = 0.005                    # step of the integration
const EVERY = 2                        # steps between two instants of the guess
const TAU   = 0.2                      # half-width of the window of the first figures
const TAUS  = (0.05, 0.1, 0.2, 0.4)    # half-widths of the windows compared, below T/2
const D     = 0.08                     # distance of the recurrence sought, relative to the state
const SPAN  = 2000                     # steps before and after a window, for the Lyapunov vectors


# ============================================================================ #
# A recurrence that does not close well                                        #
# ============================================================================ #

# ---- a trajectory on the attractor, after a transient of 20, at instants 0.01 apart ----
dt    = H * EVERY
Xfull = trajectory([1.0, 1.0, 1.0], H, 30_000)
X     = Xfull[:, 4001:EVERY:end]

# ---- the distance between x(t) and x(t + T), for T near 1.56, the period of the shortest orbit ----
ms = round(Int, 1.4 / dt):round(Int, 1.8 / dt)
is = 1:10_000
d  = [norm(X[:, i + m] .- X[:, i]) / norm(X[:, i]) for i in is, m in ms]

# ---- its local minima over segments that visit both wings, x > 0 for 25 to 75% of the time ----
wings(i, m) = 0.25 <= count(>(0), X[1, i .+ (0:m - 1)]) / m <= 0.75
minima      = [(i, j) for i in 2:length(is) - 1, j in 2:length(ms) - 1
               if d[i, j] <= minimum(d[i - 1:i + 1, j - 1:j + 1]) && wings(is[i], ms[j])]

# ---- the one closest to the distance D ----
i, j = minima[argmin([abs(d[i, j] - D) for (i, j) in minima])]
m    = ms[j]
T    = m * dt
i₀   = is[i]
@printf "recurrence at t = %.2f, period T = %.2f, relative distance %.3f\n" i₀ * dt T d[i, j]

# ---- the guess: the segment over one period, on m instants, from the step s₀ of the trajectory ----
guess = X[:, i₀ .+ (0:m - 1)]
s₀    = 4001 + (i₀ - 1) * EVERY
M     = m * EVERY


# ============================================================================ #
# The residual, by the spectral derivative over the period                     #
# ============================================================================ #

# the spectral differentiation matrix on m instants equally spaced over the period T
function derivative(m, T)
    Dm = zeros(m, m)
    for j in 1:m, l in 1:m
        j == l && continue
        x        = (j - l) * π / m
        Dm[j, l] = 0.5 * (iseven(j - l) ? 1 : -1) * (iseven(m) ? cot(x) : csc(x))
    end
    return Dm .* (2π / T)
end

# the residual dx/dt - F(x) at each instant of a periodic guess G over the period T
residual(G, T) = [norm(r) for r in eachcol(G * derivative(size(G, 2), T)' .- hcat(lorenz.(eachcol(G))...))]

# ---- the guess without bridge on instants 0.04, 0.02, 0.01 and 0.005 apart, and its jump ----
jump   = norm(Xfull[:, s₀ + M] .- Xfull[:, s₀])
coarse = [(k * H, residual(Xfull[:, s₀ .+ (0:k:M - 1)], T)) for k in (8, 4, 2, 1)]
r₀     = residual(guess, T)
@printf "jump %.3f; residual at the junction %s, against ln 2 |J| / Δt %s\n" jump join([@sprintf("%.0f", r[1]) for (_, r) in coarse], ", ") join([@sprintf("%.0f", log(2) * jump / Δ) for (Δ, _) in coarse], ", ")


# ============================================================================ #
# The covariant Lyapunov vectors                                               #
# ============================================================================ #

# one step of the tangent and of the adjoint of the scheme, along the free trajectory from x
function tangent_step(x, δx, h)
    J  = jacobian.(rk4(x, h, 0.0, 0.0, 0.0)[2])
    k1 = J[1] * δx
    k2 = J[2] * (δx .+ (h / 2) .* k1)
    k3 = J[3] * (δx .+ (h / 2) .* k2)
    k4 = J[4] * (δx .+ h .* k3)
    return δx .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function adjoint_step(x, λ, h)
    J  = jacobian.(rk4(x, h, 0.0, 0.0, 0.0)[2])
    μ4 = (h / 6) .* λ
    a4 = J[4]' * μ4
    μ3 = (h / 3) .* λ .+ h .* a4
    a3 = J[3]' * μ3
    μ2 = (h / 3) .* λ .+ (h / 2) .* a3
    a2 = J[2]' * μ2
    μ1 = (h / 6) .* λ .+ (h / 2) .* a2
    a1 = J[1]' * μ1
    return λ .+ a1 .+ a2 .+ a3 .+ a4
end

# the two columns of Q advanced by one adjoint step from x, orthonormalised again
adjoint_pair(x, Q, h) = Matrix(qr(hcat(adjoint_step(x, Q[:, 1], h), adjoint_step(x, Q[:, 2], h))).Q)[:, 1:2]

# The unstable, neutral and stable vectors at the steps j₀, ..., j₀ + N of the free trajectory. The
# unstable vector comes from the tangent equation forward from SPAN steps before, the neutral vector
# is the vector field, and the stable vector is normal to the plane of the two leading adjoint
# vectors, from the adjoint equation backward from SPAN steps after.
function lyapunov_vectors(j₀, N)

    # ---- unstable: every vector turns into it, forward in time ----
    v = normalize([1.0, 1.0, 1.0])
    for n in j₀ - SPAN:j₀ - 1
        v = normalize(tangent_step(Xfull[:, n], v, H))
    end
    unstable = Vector{Vector{Float64}}(undef, N + 1)
    for n in 0:N
        unstable[n + 1] = v
        n < N && (v = normalize(tangent_step(Xfull[:, j₀ + n], v, H)))
    end

    # ---- stable: normal to the leading adjoint plane, backward in time ----
    Q = Matrix(qr([1.0 0.3; 0.2 1.0; -0.5 0.7]).Q)[:, 1:2]
    for n in j₀ + N + SPAN - 1:-1:j₀ + N
        Q = adjoint_pair(Xfull[:, n], Q, H)
    end
    stable = Vector{Vector{Float64}}(undef, N + 1)
    for n in N:-1:0
        stable[n + 1] = normalize(cross(Q[:, 1], Q[:, 2]))
        n > 0 && (Q = adjoint_pair(Xfull[:, j₀ + n - 1], Q, H))
    end

    # ---- neutral: the vector field ----
    neutral = [normalize(lorenz(Xfull[:, j₀ + n])) for n in 0:N]

    return unstable, neutral, stable
end


# ============================================================================ #
# A Gaussian smoothing in the same window, for comparison                      #
# ============================================================================ #

# The guess G smoothed near the junction: every instant is mixed with its Gaussian average over the
# period, of standard deviation σ, with the weight w of the window of half-width τ. The smoothed guess
# equals the guess at the edges of the window, as the bridge does.
function smoothed(G, τ, σ)
    m   = size(G, 2)
    s   = [(l < m / 2 ? l : l - m) * dt for l in 0:m - 1]
    k   = [exp(-(min(q, m - q) * dt)^2 / (2σ^2)) for q in 0:m - 1]
    k ./= sum(k)
    KG  = hcat([sum(k[q + 1] .* G[:, mod1(l - q, m)] for q in 0:m - 1) for l in 1:m]...)
    return G .+ window.(s, τ)' .* (KG .- G)
end


# ============================================================================ #
# The bridge over the window of half-width τ                                   #
# ============================================================================ #

# the window (1 - |t/τ|^q)^p: flatter in the middle as q grows, less smooth at the edges as p falls
taper(q, p) = (t, τ) -> abs(t) < τ ? (1 - abs(t / τ)^q)^p : 0.0

function analyse(τ; shape=window, vectors=true)

    # ---- the window, from the state at T - τ to the state at τ ----
    K = round(Int, τ / dt)
    N = round(Int, 2τ / H)
    a = guess[:, m - K + 1]
    b = guess[:, K + 1]

    # ---- the bridge ----
    Φ, A, Aᵀ, t, c, path = window_problem(a, τ, N; shape)
    sol = bridge(Φ, A, Aᵀ, b, zeros(3, 2N + 1); tol=1e-10, cgtol=1e-6, maxiter=40)
    Φ(sol.f)
    P = path()

    # ---- the bridged guess, with the bridge at the instants of the window, and its residual ----
    bridged = copy(guess)
    for q in 0:K - 1
        bridged[:, m - K + 1 + q] = P[:, 1 + q * EVERY]
        bridged[:, 1 + q]         = P[:, 1 + N ÷ 2 + q * EVERY]
    end
    r = residual(bridged, T)

    # ---- the Gaussian smoothing in the same window, with the best width up to τ ----
    σs = filter(<=(τ), [0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
    rs = [residual(smoothed(guess, τ, σ), T) for σ in σs]
    k  = argmin(norm.(rs))

    # ---- the forcing at the steps, along the vector field and across it ----
    ts     = t[1:2:end]
    g      = [shape(t[2n - 1], τ) .* sol.f[:, 2n - 1] for n in 1:N + 1]
    F̂      = [normalize(lorenz(P[:, n])) for n in 1:N + 1]
    along  = [dot(g[n], F̂[n]) for n in 1:N + 1]
    across = [norm(g[n] .- along[n] .* F̂[n]) for n in 1:N + 1]

    # ---- the forcing across the flow in the unstable plane, along the flow, and stable ----
    angle = nothing
    clv   = nothing
    if vectors
        unstable, neutral, stable = lyapunov_vectors(4001 + (i₀ + m - K - 1) * EVERY, N)
        angle = [acosd(min(1.0, abs(dot(unstable[n], neutral[n])))) for n in 1:N + 1]
        plane = [normalize(unstable[n] .- dot(unstable[n], neutral[n]) .* neutral[n]) for n in 1:N + 1]
        clv   = [hcat(plane[n], neutral[n], stable[n]) \ g[n] for n in 1:N + 1]
    end

    return (τ       = τ,
            N       = N,
            sol     = sol,
            P       = P,
            r       = r,
            rms     = norm(r) / sqrt(m),
            rg      = rs[k],
            σ       = σs[k],
            rmsg    = norm(rs[k]) / sqrt(m),
            E       = 0.5 * sum(c .* vec(sum(abs2, sol.f; dims=1))),
            ts      = ts,
            g       = g,
            along   = along,
            across  = across,
            clv     = clv,
            angle   = angle,
            newton  = length(sol.miss) - 1,
            cg      = maximum(length.(sol.cg)) - 1)
end

results = [analyse(τ) for τ in TAUS]
main    = results[findfirst(==(TAU), TAUS)]

@printf "\n  τ   Newton  CG   miss       rms residual  Gaussian (σ)    peak ‖wf‖  energy   max |cos|  stable/peak\n"
for R in results
    @printf "%5.2f  %4d  %3d   %.1e   %8.3f     %7.3f (%.3f)   %7.2f   %8.2f   %.2f       %.3f\n" R.τ R.newton R.cg R.sol.miss[end] R.rms R.rmsg R.σ maximum(norm.(R.g)) R.E maximum(abs.(R.along) ./ max.(norm.(R.g), 1e-300)) maximum(abs(c[3]) for c in R.clv) / maximum(norm.(R.g))
end
@printf "rms residual of the guess without bridge %.3f\n" norm(r₀) / sqrt(m)


# ============================================================================ #
# Figures for the window TAU, one per panel                                    #
# ============================================================================ #

times  = (0:m - 1) .* dt
blue   = Makie.wong_colors()[1]
band   = (Makie.wong_colors()[2], 0.15)

# a figure with a single axis, saved in examples/ by save_figure
single(; kwargs...) = (fig = Figure(size=(760, 520), fontsize=15); (fig, Axis(fig[1, 1]; kwargs...)))
save_figure(fig, name) = save(joinpath(@__DIR__, "lorenz_$(name).png"), fig)

# the guess, the bridge and the marks of its ends, in the plane (x, z)
function plot_guess!(ax, P; ends=true)
    lines!(ax, guess[1, :], guess[3, :]; color=:black, linewidth=1.5, label="guess")
    lines!(ax, P[1, :], P[3, :]; color=blue, linewidth=2.5, label="bridge")
    scatter!(ax, [guess[1, end]], [guess[3, end]]; color=:red, marker=:rect, markersize=11, label="end")
    scatter!(ax, [guess[1, 1]], [guess[3, 1]]; color=:red, markersize=11, label="start")
    ends || return
    scatter!(ax, [P[1, 1]], [P[3, 1]]; color=:transparent, strokecolor=:red, strokewidth=2, marker=:rect,
             markersize=13, label="bridge start")
    scatter!(ax, [P[1, end]], [P[3, end]]; color=:transparent, strokecolor=:red, strokewidth=2,
             markersize=13, label="bridge end")
end

# the residual of the guess and of the bridged guess, with the forcing dashed and the window shaded
function plot_residual!(ax, R)
    before = R.ts .< 0
    vspan!(ax, [0, T - R.τ], [R.τ, T]; color=band)
    lines!(ax, times, max.(r₀, 1e-12); color=:black, linewidth=1.5, label="guess")
    lines!(ax, times, max.(R.rg, 1e-12); color=Makie.wong_colors()[6], linewidth=1.5,
           label=@sprintf("Gaussian smoothing, σ = %.3f", R.σ))
    lines!(ax, times, max.(R.r, 1e-12); color=blue, linewidth=2, label="with the bridge")
    lines!(ax, T .+ R.ts[before], max.(norm.(R.g[before]), 1e-12); color=:black, linestyle=:dash,
           linewidth=1.5, label="‖w f‖, the forcing")
    lines!(ax, R.ts[.!before], max.(norm.(R.g[.!before]), 1e-12); color=:black, linestyle=:dash, linewidth=1.5)
    ylims!(ax, 1e-7, 2e3)
end

# the norm of the forcing and its components along the vector field and across it
function plot_forcing!(ax, R)
    hlines!(ax, 0; color=:gray, linewidth=1)
    lines!(ax, R.ts, norm.(R.g); color=:black, linewidth=2, label="‖w f‖")
    lines!(ax, R.ts, R.across; color=blue, linewidth=2, linestyle=:dash, label="across the vector field")
    lines!(ax, R.ts, R.along; color=Makie.wong_colors()[2], linewidth=2, label="along the vector field")
end

# the forcing on the covariant Lyapunov vectors
function plot_clv!(ax, R)
    lines!(ax, R.ts, norm.(R.g); color=:black, linewidth=2, label="‖w f‖")
    lines!(ax, R.ts, [abs(c[1]) for c in R.clv]; color=Makie.wong_colors()[6], linewidth=2, linestyle=:dash,
           label="unstable, across the flow")
    lines!(ax, R.ts, [abs(c[2]) for c in R.clv]; color=Makie.wong_colors()[2], linewidth=2,
           label="neutral, along the flow")
    lines!(ax, R.ts, [abs(c[3]) for c in R.clv]; color=blue, linewidth=2, label="stable")
end

# ---- the recurrence and its jump, in the plane (x, z) ----
fig, ax = single(; title=@sprintf("a recurrence, T = %.2f, and its jump", T), xlabel="x", ylabel="z")
lines!(ax, guess[1, :], guess[3, :]; color=:black, linewidth=1.5, label="guess")
lines!(ax, [guess[1, end], guess[1, 1]], [guess[3, end], guess[3, 1]]; color=:red, linewidth=1.5, label="jump")
scatter!(ax, [guess[1, end]], [guess[3, end]]; color=:red, marker=:rect, markersize=11, label="end")
scatter!(ax, [guess[1, 1]], [guess[3, 1]]; color=:red, markersize=11, label="start")
axislegend(ax; position=:rb)
save_figure(fig, "recurrence")

# ---- the residual of the guess without bridge near the junction, at four resolutions ----
fig, ax = single(; title="the residual of the guess, near the junction", xlabel="time from the junction s",
                 ylabel="‖dx/dt − F(x)‖", xscale=log10, yscale=log10)
for (n, (Δ, r)) in enumerate(coarse)
    l = 1:length(r) ÷ 2
    scatterlines!(ax, (l .- 0.5) .* Δ, r[l]; color=cgrad(:viridis)[(n - 1) / 3 * 0.85], markersize=5,
                  label=@sprintf("Δt = %.3f", Δ))
    scatter!(ax, [Δ / 2], [log(2) * jump / Δ]; color=:red, marker=:xcross, markersize=12)
end
sline = 10 .^ range(log10(0.002), log10(0.8); length=50)
lines!(ax, sline, jump ./ (2 .* sline); color=:black, linestyle=:dash, linewidth=1.5, label="|J| / 2s")
axislegend(ax; position=:rt)
save_figure(fig, "jump")

# ---- the guess and the bridge ----
fig, ax = single(; title=@sprintf("the guess, T = %.2f, and the bridge", T), xlabel="x", ylabel="z")
plot_guess!(ax, main.P)
axislegend(ax; position=:rb)
save_figure(fig, "guess")

# ---- the residual at each instant over the period ----
fig, ax = single(; title="the residual at each instant", xlabel="time t", ylabel="‖dx/dt − F(x)‖", yscale=log10)
plot_residual!(ax, main)
axislegend(ax; position=:rt)
save_figure(fig, "residual")

# ---- the forcing against the vector field ----
fig, ax = single(; title="the forcing along the bridge", xlabel="time from the junction",
                 ylabel="norm and components of w f")
plot_forcing!(ax, main)
axislegend(ax; position=:lt)
save_figure(fig, "forcing")

# ---- the forcing on the covariant Lyapunov vectors ----
fig, ax = single(; title="the forcing on the covariant Lyapunov vectors", xlabel="time from the junction",
                 ylabel="|component| of w f")
plot_clv!(ax, main)
axislegend(ax; position=:lt)
save_figure(fig, "clv")

# ---- Newton's method ----
fig, ax = single(; title="Newton's method", xlabel="iteration", ylabel="relative to the first miss",
                 yscale=log10)
its = 0:length(main.sol.miss) - 1
scatterlines!(ax, its, main.sol.miss ./ main.sol.miss[1]; color=:black, label="miss at the end")
scatterlines!(ax, its[2:end], max.(main.sol.control[2:end] ./ main.sol.miss[1], 1e-16); color=blue,
              label="residual of the control")
axislegend(ax; position=:rt)
save_figure(fig, "newton")

# ---- the conjugate gradients at each iteration ----
fig, ax = single(; title="conjugate gradients on A Aᵀ δν = r", xlabel="iteration of the conjugate gradients",
                 ylabel="residual / first", yscale=log10)
for (k, h) in enumerate(main.sol.cg)
    scatterlines!(ax, 0:length(h) - 1, h; color=cgrad(:viridis)[(k - 1) / max(length(main.sol.cg) - 1, 1) * 0.85],
                  label="Newton iteration $(k - 1)")
end
axislegend(ax; position=:rt)
save_figure(fig, "cg")


# ============================================================================ #
# Figures for the windows TAUS, one panel per window                           #
# ============================================================================ #

# a figure with one panel per window, in two rows
function panels(plot!; ylabel="", legend=:rt, xlabel="time from the junction", kwargs...)
    fig = Figure(size=(1300, 950), fontsize=14)
    for (n, R) in enumerate(results)
        ax = Axis(fig[(n - 1) ÷ 2 + 1, (n - 1) % 2 + 1]; title=@sprintf("τ = %.2f", R.τ), xlabel, ylabel, kwargs...)
        plot!(ax, R)
        n == 1 && axislegend(ax; position=legend, labelsize=12)
    end
    return fig
end

save_figure(panels((ax, R) -> plot_guess!(ax, R.P); xlabel="x", ylabel="z", legend=:rb), "tau_guess")
# ---- the residual for all the windows on one plot, against the time from the junction ----
fig = Figure(size=(1150, 600), fontsize=15)
ax  = Axis(fig[1, 1]; title="the residual at each instant, for the four windows", xlabel="time from the junction",
           ylabel="‖dx/dt − F(x)‖", yscale=log10)
sj  = [(l < m / 2 ? l : l - m) * dt for l in 0:m - 1]
o   = sortperm(sj)
lines!(ax, sj[o], max.(r₀[o], 1e-12); color=:black, linewidth=1.5, label="guess")
for (n, R) in enumerate(results)
    c = cgrad(:viridis)[(n - 1) / (length(results) - 1) * 0.85]
    lines!(ax, sj[o], max.(R.rg[o], 1e-12); color=c, linewidth=1.2, linestyle=:dash)
    lines!(ax, sj[o], max.(R.r[o], 1e-12); color=c, linewidth=2, label=@sprintf("τ = %.2f, with the bridge", R.τ))
end
lines!(ax, [NaN], [NaN]; color=:gray, linewidth=1.2, linestyle=:dash, label="Gaussian smoothing, same window")
ylims!(ax, 1e-7, 2e3)
Legend(fig[1, 2], ax; labelsize=13, framevisible=false)
save_figure(fig, "tau_residual")
save_figure(panels(plot_forcing!; ylabel="norm and components of w f", legend=:lt), "tau_forcing")
save_figure(panels(plot_clv!; ylabel="|component| of w f", legend=:lt), "tau_clv")

# ---- the summary: residual, forcing, energy and iterations against the width ----
fig = Figure(size=(1500, 460), fontsize=14)
τs  = collect(TAUS)
ax1 = Axis(fig[1, 1]; title="the residual over the period", xlabel="τ", ylabel="root mean square", yscale=log10)
hlines!(ax1, norm(r₀) / sqrt(m); color=:black, linestyle=:dash, label="guess without bridge")
scatterlines!(ax1, τs, [R.rmsg for R in results]; color=Makie.wong_colors()[6], label="Gaussian smoothing")
scatterlines!(ax1, τs, [R.rms for R in results]; color=blue, label="with the bridge")
axislegend(ax1; position=:rt)
ax2 = Axis(fig[1, 2]; title="the forcing", xlabel="τ", yscale=log10)
scatterlines!(ax2, τs, [maximum(norm.(R.g)) for R in results]; color=:black, label="peak of ‖w f‖")
scatterlines!(ax2, τs, [R.E for R in results]; color=blue, label="E = ½∫‖f‖² dt")
axislegend(ax2; position=:rt)
ax3 = Axis(fig[1, 3]; title="the iterations", xlabel="τ", ylabel="iterations")
scatterlines!(ax3, τs, [R.newton for R in results]; color=:black, label="Newton")
scatterlines!(ax3, τs, [R.cg for R in results]; color=blue, label="conjugate gradients, at most")
ylims!(ax3, 0, nothing)
axislegend(ax3; position=:rt)
save_figure(fig, "tau_summary")


# ============================================================================ #
# The shape of the window                                                      #
# ============================================================================ #

# the windows compared, with their colours and lines: flatter in the middle, then less smooth
wong   = Makie.wong_colors()
SHAPES = [("(1 − s²)³", taper(2, 3), :black, :solid),
          ("(1 − s⁴)³", taper(4, 3), wong[1], :solid),
          ("(1 − s⁸)³", taper(8, 3), wong[3], :solid),
          ("(1 − s¹⁶)³", taper(16, 3), wong[5], :solid),
          ("(1 − s²)²", taper(2, 2), wong[2], :dash),
          ("1 − s²", taper(2, 1), wong[6], :dash),
          ("1, a box", taper(2, 0), wong[4], :dash)]

shapes = [[analyse(τ; shape, vectors=false) for τ in TAUS] for (_, shape, _, _) in SHAPES]
k₀     = findfirst(==(TAU), TAUS)

@printf "\nthe root mean square residual over the period, for the windows (columns τ = %s)\n" join(TAUS, ", ")
for ((label, _, _, _), Rs) in zip(SHAPES, shapes)
    @printf "%-12s %s   Newton %s\n" label join([@sprintf("%7.3f", R.rms) for R in Rs], " ") join([R.newton for R in Rs], ", ")
end
@printf "%-12s %s\n" "Gaussian" join([@sprintf("%7.3f", R.rmsg) for R in results], " ")

# ---- the windows ----
fig, ax = single(; title="the windows", xlabel="t / τ", ylabel="w")
for (label, shape, color, linestyle) in SHAPES
    lines!(ax, range(-1, 1; length=401), shape.(range(-1, 1; length=401), 1.0); color, linestyle, linewidth=2, label)
end
axislegend(ax; position=:cb, labelsize=12)
save_figure(fig, "weight_shapes")

# ---- the residual and the forcing for the windows, at τ = TAU ----
function plot_shapes(name, title, ylabel)
    fig = Figure(size=(1150, 600), fontsize=15)
    ax  = Axis(fig[1, 1]; title, xlabel="time from the junction", ylabel, yscale=name == "residual" ? log10 : identity)
    sj  = [(l < m / 2 ? l : l - m) * dt for l in 0:m - 1]
    o   = sortperm(sj)
    for ((label, _, color, linestyle), Rs) in zip(SHAPES, shapes)
        R = Rs[k₀]
        name == "residual" ? lines!(ax, sj[o], max.(R.r[o], 1e-12); color, linestyle, linewidth=2, label) :
                             lines!(ax, R.ts, norm.(R.g); color, linestyle, linewidth=2, label)
    end
    name == "residual" && ylims!(ax, 1e-7, 2e3)
    name == "forcing" && xlims!(ax, -1.1TAU, 1.1TAU)
    Legend(fig[1, 2], ax; labelsize=13, framevisible=false)
    save_figure(fig, "weight_" * name)
end
plot_shapes("forcing", @sprintf("the forcing, τ = %.2f", TAU), "‖w f‖")

# ---- the residual for the windows, at τ = TAU, over the period and zoomed on the window ----
function plot_shapes_residual()
    fig = Figure(size=(1150, 950), fontsize=15)
    ax1 = Axis(fig[1, 1]; title=@sprintf("the residual at each instant, τ = %.2f", TAU), xlabel="time from the junction",
               ylabel="‖dx/dt − F(x)‖", yscale=log10)
    ax2 = Axis(fig[2, 1]; title="zoom on the window", xlabel="time from the junction", ylabel="‖dx/dt − F(x)‖")
    sj  = [(l < m / 2 ? l : l - m) * dt for l in 0:m - 1]
    o   = sortperm(sj)
    lines!(ax1, sj[o], max.(r₀[o], 1e-12); color=:gray60, linewidth=1.5, label="guess")
    for ((label, _, color, linestyle), Rs) in zip(SHAPES, shapes)
        R = Rs[k₀]
        lines!(ax1, sj[o], max.(R.r[o], 1e-12); color, linestyle, linewidth=2, label)
        lines!(ax2, sj[o], R.r[o]; color, linestyle, linewidth=2)
    end
    ylims!(ax1, 1e-7, 2e3)
    xlims!(ax2, -1.1TAU, 1.1TAU)
    ylims!(ax2, 0, 1.1 * maximum(maximum(Rs[k₀].r) for Rs in shapes))
    vspan!(ax2, -TAU, TAU; color=band)
    rowsize!(fig.layout, 1, Relative(0.55))
    Legend(fig[1:2, 2], ax1; labelsize=13, framevisible=false)
    save_figure(fig, "weight_residual")
end
plot_shapes_residual()

# ---- the summary against the width of the window ----
fig = Figure(size=(1500, 480), fontsize=14)
ax1 = Axis(fig[1, 1]; title="the residual over the period", xlabel="τ", ylabel="root mean square", yscale=log10)
ax2 = Axis(fig[1, 2]; title="the peak of the forcing", xlabel="τ", ylabel="peak of ‖w f‖", yscale=log10)
ax3 = Axis(fig[1, 3]; title="Newton's method", xlabel="τ", ylabel="iterations")
scatterlines!(ax1, τs, [R.rmsg for R in results]; color=:gray30, linestyle=:dot, label="Gaussian smoothing")
for ((label, _, color, linestyle), Rs) in zip(SHAPES, shapes)
    scatterlines!(ax1, τs, [R.rms for R in Rs]; color, linestyle, label)
    scatterlines!(ax2, τs, [maximum(norm.(R.g)) for R in Rs]; color, linestyle)
    scatterlines!(ax3, τs, [R.newton for R in Rs]; color, linestyle)
end
ylims!(ax3, 0, nothing)
Legend(fig[1, 4], ax1; labelsize=12, framevisible=false)
save_figure(fig, "weight_summary")
