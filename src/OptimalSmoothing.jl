# OptimalSmoothing.jl: the control of least norm that steers a system exactly onto a target state,
# by Newton's method on the conditions of the optimum, with conjugate gradients on A Aᵀ.

module OptimalSmoothing

using LinearAlgebra
using Printf

export bridge

"""
    bridge(Φ, A, Aᵀ, b, f; tol=1e-8, maxiter=20, cgtol=1e-2, cgmaxiter=500, P=identity, io=devnull)

Return the control of least norm that steers a system exactly onto the state `b`,

    minimise ½⟨f, f⟩   subject to   Φ(f) = b,

starting from the control `f`. The problem is given by three functions.

  - `Φ(f)` returns the end state reached with the control `f`. It also fixes the point about which
    `A` and `Aᵀ` linearise, until the next call.
  - `A(δf)` returns the change of the end state due to a change `δf` of the control.
  - `Aᵀ(v)` returns the control `Aᵀv`, the transpose of `A` applied to the state `v`, such that
    `⟨v, A δf⟩ = ⟨Aᵀv, δf⟩` for every `δf` and `v`.

For a differential equation `dη/dt = N(η) + w f` over a window, from a fixed start, `Φ` integrates
it forward, `A` integrates the tangent equation forward from zero with the forcing `w δf`, and `Aᵀ`
integrates the adjoint equation backward from `v` and multiplies it by `w`.

The conditions of the optimum are `f = Aᵀν` and `Φ(f) = b`, with `ν` the multiplier of the end
condition. Newton's method solves them. Each iteration solves `A Aᵀ δν = b - Φ(f) + A (f - Aᵀν)` by
conjugate gradients, then sets `ν += δν` and `f = Aᵀν`. The change of `Aᵀ` with the control is left
out. This changes how fast the iterations converge, not where.

The iterations stop when the miss `‖Φ(f) - b‖` falls below `tol` times its first value and the
residual of the control `‖f - Aᵀν‖` below `tol` times `‖f‖`. The conjugate gradients stop when their
residual falls below `cgtol` times its first value. `P(r)` applies a preconditioner for `A Aᵀ` to a
state `r`. `io` receives one line per iteration.

States and controls are any objects with `copy`, broadcasting in place with scalars, and a real
inner product `dot`. The least norm is in the inner product of controls for which `Aᵀ` is the
transpose of `A`. The package uses `dot` on controls only to measure the residual of the control.

Returns the named tuple `(f, ν, miss, control, cg)`: the control, the multiplier, the miss and the
residual of the control at each iteration, and the relative residuals of the conjugate gradients
at each iteration.
"""
function bridge(Φ, A, Aᵀ, b, f;
                      tol::Real=1e-8,
                  maxiter::Int=20,
                    cgtol::Real=1e-2,
                cgmaxiter::Int=500,
                        P=identity,
                       io::IO=devnull)

    # ---- input checks ----
    tol > 0 ||
        throw(ArgumentError("tol must be positive"))
    cgtol > 0 ||
        throw(ArgumentError("cgtol must be positive"))

    # ---- the multiplier of the end condition, and the history ----
    ν       = copy(b)
    ν      .= 0
    miss    = Float64[]
    control = Float64[]
    cg      = Vector{Float64}[]

    for k in 0:maxiter

        # ---- forward: the miss at the end, and the point of the linearisation ----
        r   = copy(b)
        r .-= Φ(f)

        # ---- backward: the residual of the control, f - Aᵀν, with ν = 0 at the start ----
        c = copy(f)
        k > 0 && (c .-= Aᵀ(ν))

        # ---- convergence ----
        push!(miss, _norm(r))
        push!(control, _norm(c))
        @printf io "iteration %3d: miss %.3e, residual of the control %.3e\n" k miss[end] control[end]
        miss[end] <= tol * miss[1] && control[end] <= tol * _norm(f) && break
        k == maxiter && break

        # ---- the right-hand side, b - Φ(f) + A (f - Aᵀν) ----
        control[end] > 0 && (r .+= A(c))

        # ---- conjugate gradients on A Aᵀ δν = r ----
        δν, history = _cg(v -> A(Aᵀ(v)), r, P, cgtol, cgmaxiter)
        push!(cg, history)

        # ---- the new iterate ----
        ν .+= δν
        f   = Aᵀ(ν)
    end

    return (f       = f,
            ν       = ν,
            miss    = miss,
            control = control,
            cg      = cg)
end

# the norm of a state or a control, from its inner product
_norm(x) = sqrt(real(dot(x, x)))

# conjugate gradients for M x = rhs, with M symmetric and positive definite and P a preconditioner,
# from x = 0; returns x and the residuals relative to the first, stopping below tol
function _cg(M, rhs, P, tol, maxiter)

    # ---- start from zero ----
    x       = copy(rhs)
    x      .= 0
    r       = copy(rhs)
    z       = P(r)
    p       = copy(z)
    rz      = real(dot(r, z))
    r₀      = _norm(r)
    history = [1.0]

    # ---- iterations ----
    for _ in 1:maxiter
        q    = M(p)
        α    = rz / real(dot(p, q))
        x  .+= α .* p
        r  .-= α .* q
        push!(history, _norm(r) / r₀)
        history[end] <= tol && break

        # the next direction, conjugate to the previous ones
        z   = P(r)
        rz′ = real(dot(r, z))
        p  .= z .+ (rz′ / rz) .* p
        rz  = rz′
    end

    return x, history
end

end
