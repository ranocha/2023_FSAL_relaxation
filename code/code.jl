# Setup packages
import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


# Load packages
using LinearAlgebra
using DataStructures: top
using OrdinaryDiffEq, DiffEqCallbacks
using LaTeXStrings
using MuladdMacro: @muladd
using Roots: Roots
using SummationByPartsOperators
using PyPlot

#PyPlot.rc("text", usetex= true)
PyPlot.rc("font", family="serif", size=18.)
PyPlot.rc("legend", loc="best", fontsize="medium", fancybox= true, framealpha=0.5)
PyPlot.rc("lines", linewidth=2.5, markersize=10, markeredgewidth=2.5)

## Testproblems
# List of Testproblems
# - Harmonic oscillator
# - Nonlinear oscillator
# - Time dependent oscillator with bounded angular velocity
# - Conserved exponential entropy
# - Nonlinear Pendulum
# - Lotka-Volterra
# - Duffing oscillator
# - Henon Heiles
# - Kepler Problem
# - Linear Transport Fourier
# - Linear Transport DG
# - BBM equation
# - BBM with a cubic invariant

# Harmonic oscillator
function harmonic_osc_rhs!(du, u, p, t)
  du[1] = -u[2]
  du[2] =  u[1]
  return nothing
end

function harmonic_osc_solution(t)
  si, co = sincos(t)
  return [co, si]
end

function harmonic_osc_entropy(u, p)
  return sum(abs2, u)
end

function harmonic_osc_functionals(u, t, integrator)
  return harmonic_osc_functionals(u, t)
end
function harmonic_osc_functionals(u, t)
  entropy = harmonic_osc_entropy(u, nothing)
  error = norm(u - harmonic_osc_solution(t))
  return (entropy, error)
end

function harmonic_osc_setup(tspan = (0.0, 10.0))
  u0 = harmonic_osc_solution(first(tspan))
  params = (; entropy = harmonic_osc_entropy,
              functionals = harmonic_osc_functionals,
              relaxation_failed = Ref(false),
              relaxation_parameter = Ref(1.0))
  return ODEProblem(harmonic_osc_rhs!, u0, tspan, params)
end


# Nonlinear oscillator
function nonlinear_osc_rhs!(du, u, p, t)
  factor = inv(sum(abs2, u))
  du[1] = -u[2] * factor
  du[2] =  u[1] * factor
  return nothing
end

function nonlinear_osc_solution(t)
  si, co = sincos(t)
  return [co, si]
end

function nonlinear_osc_entropy(u, p)
  return sum(abs2, u)
end

function nonlinear_osc_functionals(u, t, integrator)
  return nonlinear_osc_functionals(u, t)
end
function nonlinear_osc_functionals(u, t)
  entropy = nonlinear_osc_entropy(u, nothing)
  error = norm(u - nonlinear_osc_solution(t))
  return (entropy, error)
end

function nonlinear_osc_setup(tspan = (0.0, 10.0))
  u0 = nonlinear_osc_solution(first(tspan))
  params = (; entropy = nonlinear_osc_entropy,
              functionals = nonlinear_osc_functionals,
              relaxation_failed = Ref(false),
              relaxation_parameter = Ref(1.0))
  return ODEProblem(nonlinear_osc_rhs!, u0, tspan, params)
end

## time dpenedent harmonic oscillator
function harmonic_osc_time_dependent_bounded_rhs!(du, u, p, t)
  du[1] = -(1 + 0.5 * sin(t)) * u[2]
  du[2] =  (1 + 0.5 * sin(t)) * u[1]
  return nothing
end

function harmonic_osc_time_dependent_bounded_solution(t)
  return [
    cos(1/2) * cos(t - 0.5 * cos(t)) - sin(1/2) * sin(t - 0.5 * cos(t)),
    sin(1/2) * cos(t - 0.5 * cos(t)) + cos(1/2) * sin(t - 0.5 * cos(t))
    ]
end

function harmonic_osc_time_dependent_bounded_entropy(u, p)
  return sum(abs2, u)
end

function harmonic_osc_time_dependent_bounded_functionals(u, t, integrator)
  return harmonic_osc_time_dependent_bounded_functionals(u, t)
end
function harmonic_osc_time_dependent_bounded_functionals(u, t)
  entropy = harmonic_osc_time_dependent_bounded_entropy(u, nothing)
  error = norm(u - harmonic_osc_time_dependent_bounded_solution(t))
  return (entropy, error)
end

function harmonic_osc_time_dependent_bounded_setup(tspan = (0.0, 10.0))
  u0 = harmonic_osc_time_dependent_bounded_solution(first(tspan))
  params = (; entropy = harmonic_osc_time_dependent_bounded_entropy,
              functionals = harmonic_osc_time_dependent_bounded_functionals,
              relaxation_failed = Ref(false),
              relaxation_parameter = Ref(1.0))
  return ODEProblem(harmonic_osc_time_dependent_bounded_rhs!, u0, tspan, params)
end

## Conserved Exponential Energy
function conserved_exponential_energy_rhs!(du, u, p, t)
  du[1] = -exp(u[2])
  du[2] =  exp(u[1])
  return nothing
end

function conserved_exponential_energy_solution(t)
  return [
    log(exp(1) + exp(3/2)) - log(exp(0.5) + exp((exp(0.5) + exp(1)) * t)),
    log((exp((exp(0.5) + exp(1)) * t) * (exp(0.5) + exp(1))) / (exp(0.5) + exp((exp(0.5) + exp(1)) * t)))
  ]
end

function conserved_exponential_energy_entropy(u, p)
  return exp(u[1]) + exp(u[2])
end

function conserved_exponential_energy_functionals(u, t, integrator)
  return conserved_exponential_energy_functionals(u, t)
end
function conserved_exponential_energy_functionals(u, t)
  entropy = conserved_exponential_energy_entropy(u, nothing)
  error = norm(u - conserved_exponential_energy_solution(t))
  return (entropy, error)
end

function conserved_exponential_energy_setup(tspan = (0.0, 10.0))
  u0 = conserved_exponential_energy_solution(first(tspan))
  params = (; entropy = conserved_exponential_energy_entropy,
              functionals = conserved_exponential_energy_functionals,
              relaxation_failed = Ref(false),
              relaxation_parameter = Ref(1.0))
  return ODEProblem(conserved_exponential_energy_rhs!, u0, tspan, params)
end

## Nonlinear Pendulum
function nonlinear_pendulum_rhs!(du, u, p, t)
  du[1] = -sin(u[2])
  du[2] =  u[1]
  return nothing
end

prob = ODEProblem(nonlinear_pendulum_rhs!, [1.5;1.0], (0.0, 10.0));
ref_sol_nonlinear_pendulum = solve(prob, Vern9(), abstol = 1e-14, reltol = 1e-14);

function nonlinear_pendulum_solution(t)
  return ref_sol_nonlinear_pendulum(t)
end

function nonlinear_pendulum_entropy(u, p)
  return 0.5 * u[1]^2 - cos(u[2])
end

function nonlinear_pendulum_functionals(u, t, integrator)
  return nonlinear_pendulum_functionals(u, t)
end
function nonlinear_pendulum_functionals(u, t)
  entropy = nonlinear_pendulum_entropy(u, nothing)
  error = norm(u - nonlinear_pendulum_solution(t))
  return (entropy, error)
end

function nonlinear_pendulum_setup(tspan = (0.0, 10.0))
  u0 = nonlinear_pendulum_solution(first(tspan))
  params = (; entropy = nonlinear_pendulum_entropy,
              functionals = nonlinear_pendulum_functionals,
              relaxation_failed = Ref(false),
              relaxation_parameter = Ref(1.0))
  return ODEProblem(nonlinear_pendulum_rhs!, u0, tspan, params)
end

## Lotka Volterra
function lotka_volterra_rhs!(du, u, p, t)
  du[1] = u[1] * (one(u[2]) - u[2])
  du[2] = u[2] * (u[1] - one(u[1]))

  return nothing
end

prob = ODEProblem(lotka_volterra_rhs!, [0.8;0.3], (0.0, 10.0));
ref_sol_lotka_volterra = solve(prob, Vern9(), abstol = 1e-14, reltol = 1e-14);

function lotka_volterra_solution(t)
  return ref_sol_lotka_volterra(t)
end

function lotka_volterra_entropy(u, p)
  return u[1] - log(u[1]) + u[2] - log(u[2])
end

function lotka_volterra_functionals(u, t, integrator)
  return lotka_volterra_functionals(u, t)
end

function lotka_volterra_functionals(u,t)
  entropy = lotka_volterra_entropy(u, nothing)
  error = norm(u - lotka_volterra_solution(t))

  return (entropy, error)
end

function lotka_volterra_setup(tspan = (0.0, 10.0))
  u0 = lotka_volterra_solution(first(tspan))
  params = (; entropy = lotka_volterra_entropy,
              functionals = lotka_volterra_functionals,
              relaxation_failed = Ref(false),
              relaxation_parameter = Ref(1.0))
  return ODEProblem(lotka_volterra_rhs!, u0, tspan, params)
end


# Duffing Oscillator
function duffing_osc_rhs!(du, u, p, t)
    du[1] = u[2] - u[2]^3
    du[2] = u[1]

    return nothing
end

prob = ODEProblem(duffing_osc_rhs!, [0.0; 1.14142], (0.0, 10.0));
ref_sol_duffing_osc = solve(prob, Vern9(), abstol = 1e-14, reltol = 1e-14);

function duffing_osc_solution(t)

    return ref_sol_duffing_osc(t)
end

function duffing_osc_entropy(u, p)
    return 0.5 * u[1]^2 - 0.5 * u[2]^2 + 0.25 * u[2]^4
end

function duffing_osc_functionals(u, t, integrator)
    return duffing_osc_functionals(u, t)
end

function duffing_osc_functionals(u, t)
    entropy = duffing_osc_entropy(u, nothing)
    error = norm(u - duffing_osc_solution(t))
    return (entropy, error)
end

function duffing_osc_setup(tspan = (0.0, 10.0))
    u0 = duffing_osc_solution(first(tspan))
    params = (; entropy = duffing_osc_entropy,
                functionals = duffing_osc_functionals,
                relaxation_failed = Ref(false),
                relaxation_parameter = Ref(1.0))
    return ODEProblem(duffing_osc_rhs!, u0, tspan, params)
end


# Henon Heiles
function henon_heiles_rhs!(du, u, p, t)
    du[1] = -u[3]  - 2 * u[3] * u[4]
    du[2] = -u[4] - u[3]^2 + u[4]^2
    du[3] = u[1]
    du[4] = u[2]

    return nothing
end

prob = ODEProblem(henon_heiles_rhs!, [0.12; 0.12; 0.12; 0.12], (0.0, 10.0));
ref_sol_henon_heiles = solve(prob, Vern9(), abstol = 1e-14, reltol = 1e-14);

function henon_heiles_solution(t)
    return ref_sol_henon_heiles(t)
end

function henon_heiles_entropy(u, p)
    return (u[1]^2 + u[2]^2)/2 + (u[3]^2 + u[4]^2)/2 + u[3]^2 * u[4] - u[4]^3/3
end

function henon_heiles_functionals(u, t, integrator)
    return henon_heiles_functionals(u, t)
end

function henon_heiles_functionals(u, t)
    entropy = henon_heiles_entropy(u, nothing)
    error = norm(u - henon_heiles_solution(t))
    return (entropy, error)
end

function henon_heiles_setup(tspan = (0.0, 10.0))
    u0 = henon_heiles_solution(first(tspan))
    params = (; entropy = henon_heiles_entropy,
                functionals = henon_heiles_functionals,
                relaxation_failed = Ref(false),
                relaxation_parameter = Ref(1.0))
    return ODEProblem(henon_heiles_rhs!, u0, tspan, params)
end


# Kepler Problem
function kepler_rhs!(du, u, p, t)
    denominator = sqrt(u[3]^2 + u[4]^2)^3
    du[1] = -u[3] / denominator
    du[2] = -u[4] / denominator
    du[3] = u[1]
    du[4] = u[2]

    return nothing
end

prob = ODEProblem(kepler_rhs!, [0.0, sqrt((1.0 + 0.6)/((1.0 - 0.6))), 1.0 - 0.6, 0.0], (0.0, 10.0));
ref_sol_kepler = solve(prob, Vern9(), abstol = 1e-14, reltol = 1e-14);

function kepler_solution(t)
    return ref_sol_kepler(t)
end

function kepler_entropy(u, p)
    return 0.5 * (u[1]^2 + u[2]^2) - 1 / sqrt(u[3]^2 + u[4]^2)
end

function kepler_functionals(u, t, integrator)
    return kepler_functionals(u, t)
end

function kepler_functionals(u, t)
    entropy = kepler_entropy(u, nothing)
    error = norm(u - kepler_solution(t))
    return (entropy, error)
end

function kepler_setup(tspan = (0.0, 10.0))
    u0 = kepler_solution(first(tspan))
    params = (; entropy = kepler_entropy,
                functionals = kepler_functionals,
                relaxation_failed = Ref(false),
                relaxation_parameter = Ref(1.0))
    return ODEProblem(kepler_rhs!, u0, tspan, params)
end


# Linear transport DG
function linear_transport_DG_rhs!(du, u, param, t)
  (; D) = param
  mul!(du, D, u)
  @. du = -du

  return nothing
end

function linear_transport_DG_entropy(u, param)
  (; D) = param
  return integrate(abs2, u, D)
end

function linear_transport_DG_functionals(u, t, integrator)
  # Functionals to save during the time integration
  (; D, tmp1, L) = integrator.p
  x = SummationByPartsOperators.grid(D)

  lt_entropy = linear_transport_DG_entropy(u, integrator.p)

  @. tmp1 = u - linear_transport_DG_solution(t, x; L)
  error_l2 = integrate(abs2, tmp1, D) |> sqrt

  return (lt_entropy, error_l2)
end

function linear_transport_DG_solution(t, x; L = 1.0)
  #we use the function exp(sin(2 * pi / L * x)) as initial_condition
  return exp(sin(2 * pi / L * (x - t)))
end

function linear_transport_DG_setup()
  xmin = 0.0
  xmax =  2.0
  L = xmax - xmin
  nelements = 8
  polydeg = 5
  mesh = UniformPeriodicMesh1D(; xmin, xmax, Nx = nelements)
  D = couple_discontinuously(
    legendre_derivative_operator(xmin = -1.0, xmax = 1.0, N = polydeg + 1),
    mesh,
    Val{:central}() # central numerical flux for linear advection
  )

  tspan = (0.0, 100.0)
  x = SummationByPartsOperators.grid(D)
  u0 = linear_transport_DG_solution.(tspan[1], x; L)
  tmp1 = similar(u0)

  param = (; D, linear_transport_DG_solution, tmp1, L,
             entropy = linear_transport_DG_entropy,
             functionals = linear_transport_DG_functionals,
             relaxation_failed = Ref(false),
             relaxation_parameter = Ref(1.0))

  return ODEProblem(linear_transport_DG_rhs!, u0, tspan, param)
end

## BBM equation
function bbm_rhs!(du, u, param, t)
  (; D1, invImD2, tmp1, tmp2) = param
  one_third = one(t) / 3

  # this semidiscretization conserves the linear and quadratic invariants
  @. tmp1 = -one_third * u^2
  mul!(tmp2, D1, tmp1)
  mul!(tmp1, D1, u)
  @. tmp2 += -one_third * u * tmp1 - tmp1
  ldiv!(du, invImD2, tmp2)

  return nothing
end
function bbm_solution(t, x)
  # Physical setup of a traveling wave solution with speed `c`
  xmin = -90.0
  xmax =  90.0
  c = 1.2

  A = 3 * (c - 1)
  K = 0.5 * sqrt(1 - 1 / c)
  x_t = mod(x - c * t - xmin, xmax - xmin) + xmin

  return A / cosh(K * x_t)^2
end

function bbm_entropy(u, param)
  (; D1, D2, tmp1) = param

  mul!(tmp1, D2, u)
  @. tmp1 = u^2 - u * tmp1
  quadratic = integrate(tmp1, D1)

  return quadratic
end

function bbm_functionals(u, t, integrator)
  # Functionals to save during the time integration
  (; D1, tmp1) = integrator.p
  x = SummationByPartsOperators.grid(D1)

  # linear = integrate(u, D1)

  quadratic = bbm_entropy(u, integrator.p)

  # @. tmp1 = (u + 1)^3
  # cubic = integrate(tmp1, D1)

  @. tmp1 = u - bbm_solution(t, x)
  error_l2 = integrate(abs2, tmp1, D1) |> sqrt

  # return (linear, quadratic, cubic, error_l2)
  return (quadratic, error_l2)
end

function bbm_setup(; domain_traversals = 1)
  nnodes = 2^8
  xmin = -90.0
  xmax =  90.0
  c = 1.2

  D1 = fourier_derivative_operator(xmin, xmax, nnodes)
  D2 = D1^2
  invImD2 = I - D2

  tspan = (0.0, (xmax - xmin) / (3 * c) + domain_traversals * (xmax - xmin) / c)
  x = SummationByPartsOperators.grid(D1)
  u0 = bbm_solution.(tspan[1], x)
  tmp1 = similar(u0)
  tmp2 = similar(u0)
  param = (; D1, D2, invImD2, tmp1, tmp2, bbm_solution,
             entropy = bbm_entropy,
             functionals = bbm_functionals,
             relaxation_failed = Ref(false),
             relaxation_parameter = Ref(1.0))

  return ODEProblem(bbm_rhs!, u0, tspan, param)
end

## BBM cubic

# semidiscretization of the BBM equation conserving the cubic invariant
function bbm_cubic_rhs!(du, u, param, t)
  (; D1, invImD2, tmp1, tmp2) = param
  one_half = one(t) / 2

  # this semidiscretization conserves the linear and cubic invariants
  @. tmp1 = -(one_half * u^2 + u)
  mul!(tmp2, D1, tmp1)
  ldiv!(du, invImD2, tmp2)

  return nothing
end

function bbm_cubic_entropy(u, param)
  (; D1, tmp1) = param

  @. tmp1 = (u + 1)^3
  cubic = integrate(tmp1, D1)

  return cubic
end

function bbm_cubic_functionals(u, t, integrator)
  # Functionals to save during the time integration
  (; D1, tmp1) = integrator.p
  x = SummationByPartsOperators.grid(D1)

  # linear = integrate(u, D1)

  cubic = bbm_cubic_entropy(u, integrator.p)

  @. tmp1 = u - bbm_solution(t, x)
  error_l2 = integrate(abs2, tmp1, D1) |> sqrt

  return (cubic, error_l2)
end

function bbm_cubic_setup(; domain_traversals = 1)
  nnodes = 2^8
  xmin = -90.0
  xmax =  90.0
  c = 1.2

  D1 = fourier_derivative_operator(xmin, xmax, nnodes)
  D2 = D1^2
  invImD2 = I - D2

  tspan = (0.0, (xmax - xmin) / (3 * c) + domain_traversals * (xmax - xmin) / c)
  x = SummationByPartsOperators.grid(D1)
  u0 = bbm_solution.(tspan[1], x)
  tmp1 = similar(u0)
  tmp2 = similar(u0)
  param = (; D1, D2, invImD2, tmp1, tmp2, bbm_solution,
             entropy = bbm_cubic_entropy,
             functionals = bbm_cubic_functionals,
             relaxation_failed = Ref(false),
             relaxation_parameter = Ref(1.0))

  return ODEProblem(bbm_cubic_rhs!, u0, tspan, param)
end


# Plotting helper function
function plot_kwargs()
  fontsizes = (
    xtickfontsize = 14, ytickfontsize = 14,
    xguidefontsize = 16, yguidefontsize = 16,
    legendfontsize = 14)
  (; linewidth = 3, gridlinewidth = 2,
     markersize = 8, markerstrokewidth = 4,
     fontsizes...)
end


# Relaxation
struct RelaxationFunctional{U, P, F, T}
  utmp::U
  unew::U
  uold::U
  param::P
  entropy::F
  entropy_old::T
end

function (f::RelaxationFunctional)(γ)
  @. f.utmp = f.uold + γ * (f.unew - f.uold)
  return f.entropy(f.utmp, f.param) - f.entropy_old
end

function relaxation!(integrator::OrdinaryDiffEq.ODEIntegrator; relaxation_at_last_step = true)
  #we use this function to update our solution to the relaxation parameter
  told = integrator.tprev
  uold = integrator.uprev
  tnew = integrator.t
  unew = integrator.u
  utmp = first(get_tmp_cache(integrator))
  entropy = integrator.p.entropy

  γ = integrator.p.relaxation_parameter[]

  next_tstop = top(integrator.opts.tstops)
  if !(tnew ≈ next_tstop)
    # Now, the current time is not approximately equal to the next tstop

    tγ = told + γ * (tnew - told)
    # We should not step past the final time
    if tγ < next_tstop
      DiffEqBase.set_t!(integrator, tγ)
      @. unew = uold + γ * (unew - uold)
      DiffEqBase.set_u!(integrator, unew)
      DiffEqBase.u_modified!(integrator, true)
    elseif (tγ >= next_tstop) && relaxation_at_last_step
      tγ = next_tstop
      DiffEqBase.set_t!(integrator, tγ)
      @. unew = uold + γ * (unew - uold)
      DiffEqBase.set_u!(integrator, unew)
      DiffEqBase.u_modified!(integrator, true)
    else
      DiffEqBase.u_modified!(integrator, false)
    end
  else
    # Now, the current time is approximately equal to the next tstop.
    if relaxation_at_last_step
      # We use relaxation but do not change the time from the final time
      @. unew = uold + γ * (unew - uold)
      DiffEqBase.set_u!(integrator, unew)
      DiffEqBase.u_modified!(integrator, true)
    else
      DiffEqBase.u_modified!(integrator, false)
    end
  end
  #=
  if terminate_integration
    terminate!(integrator)
  end
  =#
  return nothing
end

function compute_relaxation_parameter!(u, integrator, p, t)
  #we use this function to update our solution to the relaxation parameter
  told = integrator.tprev
  uold = integrator.uprev
  tnew = integrator.t
  unew = integrator.u
  utmp = first(get_tmp_cache(integrator))
  entropy = integrator.p.entropy

  γ = one(tnew)
  terminate_integration = false
  γlo = (4//5) * one(γ)
  γhi = (6//5) * one(γ)
  functional_old = entropy(uold, integrator.p)

  functional = RelaxationFunctional(utmp, unew, uold, integrator.p,
                                    integrator.p.entropy, functional_old)
  functional_lo = functional(γlo)
  functional_hi = functional(γhi)
  if functional_lo * functional_hi > 0

    if abs(functional_lo) + abs(functional_hi) < 10 * eps()
      integrator.p.relaxation_parameter[] = one(tnew)
      integrator.p.relaxation_failed[] = false
    else
      #@warn "Terminating integration since no suitable relaxation parameter can be found"
      integrator.p.relaxation_failed[] = true
    end
  else
    integrator.p.relaxation_parameter[]  = Roots.find_zero(functional, (γlo, γhi), Roots.AlefeldPotraShi())
    integrator.p.relaxation_failed[] = false
  end
  if γ < eps(typeof(γ))
      #@warn "Terminating integration since no suitable relaxation parameter can be found"
      integrator.p.relaxation_failed[] = true
  end
end

function relaxation_failed(u,p,t)
  return p.relaxation_failed[]
end

# Use OrdinaryDiffEq.jl with relaxation
function integrate_OrdinaryDiffEq(ode, alg, tol; relaxation = true, dt = 0.0, relaxation_at_last_step = true, kwargs...)
  ode.p.relaxation_failed[] = false

  # Prepare callbacks
  saved_values = SavedValues(eltype(ode.tspan), NTuple{2, eltype(ode.u0)})
  saving = SavingCallback(ode.p.functionals, saved_values)
  if relaxation

    relaxation = DiscreteCallback(
      (u, t, integrator) -> true,
      integrator -> relaxation!(integrator; relaxation_at_last_step),
      save_positions = (false, false))

    callbacks = CallbackSet(relaxation, saving)
  else
    callbacks = saving
  end

  controller = default_controller(alg)

  if iszero(dt)
    dt, _ = ode_determine_initdt(
      ode.u0, first(ode.tspan), tol, tol, ode, OrdinaryDiffEq.alg_order(alg))
  end
  sol = solve(ode, alg; dt, abstol = tol, reltol = tol,
                    save_everystep = false,
                    callback = callbacks,
                    isoutofdomain = relaxation_failed,
                    controller, kwargs...)
  return (; t = saved_values.t,
            entropy = first.(saved_values.saveval),
            error = last.(saved_values.saveval),
            nf = sol.destats.nf,
            naccept = sol.destats.naccept,
            nreject = sol.destats.nreject,
            sol,)
end


# Structures for manual time integration loops
struct ButcherTableau{T}
  A::Matrix{T}
  b::Vector{T}
  bembd::Vector{T}
  bdiff::Vector{T}
  c::Vector{T}
  order::Int
end

function ButcherTableau(A, b, bembd, c, order)
  bdiff = b - bembd
  return ButcherTableau(A, b, bembd, bdiff, c, order)
end

function ButcherTableau(alg::BS3)
  A = [0 0 0 0; 1/2 0 0 0; 0 3/4 0 0; 2/9 1/3 4/9 0]
  b = A[end, :]
  bembd = [7/24, 1/4, 1/3, 1/8]
  c = [0, 1/2, 3/4, 1]
  order = 3
  return ButcherTableau(A, b, bembd, c, order)
end
default_controller(alg::BS3) = PIDController(0.6, -0.2)

function ButcherTableau(alg::DP5)
  A = [0 0 0 0 0 0 0;
       1/5 0 0 0 0 0 0;
       3/40 9/40 0 0 0 0 0;
       44/45 -56/15 32/9 0 0 0 0;
       19372/6561 -25360/2187 64448/6561 -212/729 0 0 0;
       9017/3168 -355/33 46732/5247 49/176 -5103/18656 0 0;
       35/384 0 500/1113 125/192 -2187/6784 11/84 0]
  b = A[end, :]
  bembd = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
  c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
  order = 5
  return ButcherTableau(A, b, bembd, c, order)
end
default_controller(alg::DP5) = PIDController(0.7, -0.4)

function ButcherTableau(alg::RK4)
  A = [0 0 0 0 0;
       1/2 0 0 0 0;
       0 1/2 0 0 0;
       0 0 1 0 0
       1/6 1/3 1/3 1/6 0]

  b = A[end, :]
  bembd = zero(b)
  c = [0, 1/2, 1/2, 1, 1]
  order = 4
  return ButcherTableau(A, b, bembd, c, order)
end
default_controller(alg::RK4) = PIDController(0.7, -0.4)

function ButcherTableau(alg::BS5)
  A = [0 0 0 0 0 0 0 0;
       1/6 0 0 0 0 0 0 0;
       2/27 4/27 0 0 0 0 0 0;
       183/1372 -162/343 1053/1372 0 0 0 0 0;
       68/297 -4/11 42/143 1960/3861 0 0 0 0;
       597/22528 81/352 63099/585728 58653/366080 4617/20480 0 0 0;
       174197/959244 -30942/79937 8152137/19744439 666106/1039181 -29421/29068 482048/414219 0 0;
       587/8064 0 4440339/15491840 24353/124800 387/44800 2152/5985 7267/94080 0]

  b = A[end, :]
  bembd = [2479/34992, 0, 123/416, 612941/3411720, 43/1440, 2272/6561, 79937/1113912, 3293/556956]
  c = [2479/34992, 0, 123/416, 612941/3411720, 43/1440, 2272/6561, 79937/1113912, 3293/556956]
  order = 5

  return ButcherTableau(A, b, bembd, c, order)
end
default_controller(alg::BS5) = PIDController(0.28, -0.23)

function ButcherTableau(alg::Tsit5)
  A = [0 0 0 0 0 0 0;
       1.61e-1 0 0 0 0 0 0;
       -8.48065549e-3 3.35480655e-1 0 0 0 0 0;
       2.89715306 -6.35944849 4.36229543 0 0 0 0;
       5.32586483 -1.17488836 7.49553934 -9.24950664e-2 0 0 0;
       5.86145544 -1.29209693e+1 8.15936790 -7.15849733e-2 -2.82690504e-2 0 0;
       9.64607668e-2 1.0e-2 4.79889650e-1 1.37900857 -3.29006952 2.32471052 0]


  #b = [0.09646077, 0.01, 0.47988965, 1.37900857, -3.29006952, 2.32471052, 0]
  b = A[end, :]
  bembd = [0.09468076, 0.00918357, 0.48777053, 1.23429757, -2.70771235, 1.86662842, 0.01515152]
  c = [0.0, 0.161, 0.327, 0.9, 0.98, 1.0, 1.0]
  order = 5
  return ButcherTableau(A, b, bembd, c, order)
end
default_controller(alg::Tsit5) = PIDController(0.7, -0.4)

function ButcherTableau(alg::RDPK3SpFSAL49)
  A = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      0.2836343005184365 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      -0.42524235346518496 0.9736500104654742 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      0.34947104589584765 -0.3189840922531554 0.33823592252425155 0.0 0.0 0.0 0.0 0.0 0.0 0.0
      -1.1454601466623768 1.9740707774514532 -1.150728913692543 -0.35849436111061833 0.0 0.0 0.0 0.0 0.0 0.0
      0.11056927920649595 0.13816563826520348 0.09617578616276438 0.0110558528570782 -0.004113944068471528 0.0 0.0 0.0 0.0 0.0
      -0.005576592812998342 0.2688173132870521 -0.017353055461134675 -0.020314507097241002 0.012399942915328739 1.4279688940485864 0.0 0.0 0.0 0.0
      0.0599230079816895 0.2550398960928815 0.04395507975525481 -0.006914765525303224 0.005735347094610085 0.5957046547103957 0.01808470948394314 0.0 0.0 0.0
      0.05048130569255466 0.20675215483652512 0.03739667048761353 -0.005244784796996848 0.004473445716314634 0.46075795449468565 -0.010036827009418115 0.1605770645946802 0.0 0.0
      0.045037326272637464 0.18592173036998483 0.03329729672569716 -0.004784204180958976 0.004055835961031311 0.41850277725960744 -0.004381901968919326 0.0271284379644609 0.2952227015964592 0.0]

  b = A[end, :]
  bembd = [2.483675912451591196775756814283216443e-02,
           1.866327774562103796990092260942180726e-01,
           5.671080795936984495604436622517631183e-02,
           -3.447695439149287702616943808570747099e-03,
           3.602245056516636472203469198006404016e-03,
           4.545570622145088936800484247980581766e-01,
           -2.434665289427612407531544765622888855e-04,
           6.642755361103549971517945063138312147e-02,
           1.613697079523505006226025497715177578e-01,
           4.955424859358438183052504342394102722e-02]
  c = vec(sum(A, dims = 2))
  order = 4

  return ButcherTableau(A, b, bembd, c, order)
end
default_controller(alg::RDPK3SpFSAL49) = PIDController(0.38, -0.18, 0.01)

# This struct contains all information required for the simple time stepping
# and mimics the approach of OrdinaryDiffEq at the same time but with
# significantly reduced complexity.
mutable struct Integrator{uType, tType, Prob, RealT, Controller, Callback}
  t::tType
  dt::tType
  u::uType
  uembd::uType
  uprev::uType
  utmp::Vector{uType}
  ktmp::Vector{uType}
  fsalfirst::uType
  prob::Prob
  tableau::ButcherTableau{RealT}
  abstol::RealT
  reltol::RealT
  controller::Controller
  naccept::Int
  nreject::Int
  nf::Int
  relaxation::Bool
  relaxation_at_last_step::Bool
  adaptive::Bool
  callback::Callback
end

function Base.getproperty(integrator::Integrator, name::Symbol)
  name === :p && return getfield(integrator, :prob).p
  return getfield(integrator, name)
end

function OrdinaryDiffEq.get_tmp_cache(integrator::Integrator)
  return integrator.utmp
end

function OrdinaryDiffEq.u_modified!(integrator::Integrator, modified::Bool)
  if modified
    prob = integrator.prob
    prob.f(integrator.fsalfirst, integrator.u, prob.p, first(prob.tspan))
    integrator.nf += 1
  end
end

# A simple function applying the explicit Runge-Kutta method defined by
# `coefficients` to tsolve the ODE `prob`.
function ode_solve(prob::ODEProblem, tableau, solve!;
                   dt = 0.0, abstol, reltol, controller, relaxation, relaxation_at_last_step, adaptive, callback, kwargs...)

  # initialization
  t = first(prob.tspan)
  u = copy(prob.u0)
  uprev = copy(u)
  uembd = similar(u)
  utmp = Vector{typeof(u)}(undef, length(tableau.b))
  for i in eachindex(utmp)
    utmp[i] = similar(u)
  end
  ktmp = Vector{typeof(u)}(undef, length(tableau.b))
  for i in eachindex(ktmp)
    ktmp[i] = similar(u)
  end
  fsalfirst = similar(u)
  naccept = 0
  nreject = 0
  nf = 0

  if iszero(dt)
    dt, fsalfirst = ode_determine_initdt(
      prob.u0, first(prob.tspan), abstol, reltol, prob, tableau.order)
  else
    prob.f(fsalfirst, prob.u0, prob.p, first(prob.tspan))
    nf += 1
  end
  integrator = Integrator(t, dt, u, uembd, uprev, utmp, ktmp, fsalfirst, prob,
                          tableau, abstol, reltol, controller,
                          naccept, nreject, nf, relaxation, relaxation_at_last_step, adaptive, callback)

  # main loop
  solve!(integrator; kwargs...)

  return (; t = [first(prob.tspan), integrator.t],
            u = [copy(prob.u0), integrator.u],
            destats = (; nf = integrator.nf,
                         naccept = integrator.naccept,
                         nreject = integrator.nreject, ))
end


function compute_error_estimate(u, uprev, uembd, abstol, reltol)
  err = zero(eltype(u))
  err_n = 0
  for i in eachindex(u)
    tol = abstol + reltol * max(abs(u[i]), abs(uprev[i]))
    if tol > 0
      err += (abs(u[i] - uembd[i]) / tol)^2
      err_n += 1
    end
  end
  return sqrt(err / err_n)
end

#=
A PID controller is of the form
struct PIDController{QT, Limiter} <: AbstractController
  beta::Vector{QT}  # controller coefficients (length 3)
  err ::Vector{QT}  # history of the error estimates (length 3)
  accept_safety::QT # accept a step if the predicted change of the step size
                    # is bigger than this parameter
  limiter::Limiter  # limiter of the dt factor (before clipping)
end
with
default_dt_factor_limiter(x) = one(x) + atan(x - one(x))
as default `limiter`, `accept_safety=0.81` as default value. The vector `beta`
contains the coefficients βᵢ of the PID controller. The vector `err` stores
error estimates.
=#
function compute_dt_factor!(controller::PIDController, error_estimate, order_for_control)
  beta1, beta2, beta3 = controller.beta
  error_estimate_min = eps(typeof(error_estimate))
  error_estimate = ifelse(error_estimate > error_estimate_min, error_estimate, error_estimate_min)

  controller.err[1] = inv(error_estimate)
  err1, err2, err3 = controller.err

  k = order_for_control
  dt_factor = err1^(beta1 / k) * err2^(beta2 / k) * err3^(beta3 / k)


  if isnan(dt_factor)
    @warn "unlimited dt_factor" dt_factor err1 err2 err3 beta1 beta2 beta3 k controller.err[1] controller.err[2] controller.err[3]
    error()
  end
  dt_factor = controller.limiter(dt_factor)
  return dt_factor
end

function accept_step(controller::PIDController, dt_factor)
  return dt_factor >= controller.accept_safety
end

function accept_step!(controller::PIDController)
  controller.err[3] = controller.err[2]
  controller.err[2] = controller.err[1]
  return nothing
end

function reject_step!(controller::PIDController)
  return nothing
end

@muladd function ode_determine_initdt(
      u0, t, abstol, reltol,
      prob::DiffEqBase.AbstractODEProblem{uType, tType},
      order) where {tType, uType}
  tdir = true
  _tType = eltype(tType)
  f = prob.f
  p = prob.p
  oneunit_tType = oneunit(_tType)
  dtmax = last(prob.tspan) - first(prob.tspan)
  dtmax_tdir = tdir * dtmax

  dtmin = nextfloat(OrdinaryDiffEq.DiffEqBase.prob2dtmin(prob))
  smalldt = convert(_tType, oneunit_tType * 1 // 10^(6))

  sk = @. abstol + abs(u0) * reltol


  f₀ = similar(u0)
  f(f₀, u0, p, t)

  tmp = @. u0 / sk
  d₀ = norm(tmp)

  @. tmp = f₀ / sk * oneunit_tType
  d₁ = norm(tmp)

  if d₀ < 1//10^(5) || d₁ < 1//10^(5)
    dt₀ = smalldt
  else
    dt₀ = convert(_tType,oneunit_tType*(d₀/d₁)/100)
  end
  dt₀ = min(dt₀, dtmax_tdir)

  if typeof(one(_tType)) <: AbstractFloat && dt₀ < 10eps(_tType) * oneunit(_tType)
    return tdir * smalldt
  end

  dt₀_tdir = tdir * dt₀

  u₁ = zero(u0)
  @. u₁ = u0 + dt₀_tdir * f₀
  f₁ = zero(f₀)
  f(f₁, u₁, p, t + dt₀_tdir)

  f₀ == f₁ && return tdir * max(dtmin, 100 * dt₀)

  @. tmp = (f₁ - f₀) / sk * oneunit_tType
  d₂ = norm(tmp) / dt₀ * oneunit_tType

  max_d₁d₂ = max(d₁, d₂)
  if max_d₁d₂ <= 1 // Int64(10)^(15)
    dt₁ = max(convert(_tType, oneunit_tType * 1 // 10^(6)), dt₀ * 1 // 10^(3))
  else
    dt₁ = convert(_tType,
                  oneunit_tType *
                  10.0^(-(2 + log10(max_d₁d₂)) / order))
  end
  return tdir * max(dtmin, min(100 * dt₀, dt₁, dtmax_tdir)), f₀
end


# Naive combination of FSAL and relaxation
@muladd function solve_naive!(integrator::Integrator)
  (; u, uprev, uembd, utmp, ktmp, fsalfirst,
     tableau, abstol, reltol, controller, relaxation, callback) = integrator
  (; A, b, bembd, c, order) = tableau
  f! = integrator.prob.f.f
  params = integrator.prob.p
  tend = last(integrator.prob.tspan)

  relaxation_at_last_step = integrator.relaxation_at_last_step
  adaptive = integrator.adaptive

  accept_step_flag = true
  relaxation_flag = true
  # solve until we have reached the final time
  while integrator.t < tend
    # adjust `dt` at the final time
    if integrator.t + integrator.dt > tend
      integrator.dt = tend - integrator.t
    end

    # compute explicit RK step with step size `dt`
    let i = 1 # first stage is special
      copy!(utmp[i], uprev)
      @. ktmp[i] = fsalfirst
    end
    for i in 2:length(b)
      copy!(utmp[i], uprev)
      for j in 1:i-1
        @. utmp[i] = utmp[i] + (A[i,j] * integrator.dt) * ktmp[j]
      end
      f!(ktmp[i], utmp[i], params, integrator.t + c[i] * integrator.dt)
      integrator.nf += 1
    end
    copy!(u,     uprev)
    copy!(uembd, uprev)
    for i in eachindex(b)
      @. u     = u     + (b[i]     * integrator.dt) * ktmp[i]
      @. uembd = uembd + (bembd[i] * integrator.dt) * ktmp[i]
    end
    #check if we want stepsizecontrol or not
    if adaptive
      # compute error estimate
      error_estimate = compute_error_estimate(u, uprev, uembd, abstol, reltol)

      # adapt `dt`
      dt_factor = compute_dt_factor!(controller, error_estimate, order)
      accept_step_flag = accept_step(controller, dt_factor)
    end

    # accept or reject the step and update `dt`
    if accept_step_flag
      # @info "accept" integrator.t
      # We don't call `accept_step!(controller)` here
      # since we may reject the step later if relaxation fails.
      #check if we have reached the final time and we have to perform relaxation at that PIDController
      if !relaxation_at_last_step && integrator.t + integrator.dt == tend
        relaxation = false
      end
      # perform relaxation
      if relaxation
        told = integrator.t
        uold = uprev
        tnew = integrator.t + integrator.dt
        unew = u
        temp = first(get_tmp_cache(integrator))
        entropy = params.entropy

        γ = one(tnew)
        terminate_integration = false
        γlo = (4//5) * one(γ)
        γhi = (6//5) * one(γ)
        functional_old = entropy(uold, params)

        functional = RelaxationFunctional(temp, unew, uold, params,
                                          params.entropy, functional_old)
        functional_lo = functional(γlo)
        functional_hi = functional(γhi)
        #first we check if we are able to perform relaxation with the values computed above
        if functional_lo * functional_hi > 0
          #in order to compute a root numerically we need different signs for functional_hi and functional_lo

          #If both values are close to machine precision anyway we make an exception from above rule,
          #since the relaxation parameter is defined as a root of a nonlinear equation
          if abs(functional_lo) + abs(functional_hi) < 10 * eps()
            #since surrounding values are close to machine precision,
            #we kind of have a free choice
            #in this case we choose γ = 1.0 since theory suggests that the relaxation parameter is close to 1 anyway
            #since the relaxation parameter is initiated with 1 we do not have to take any further actions

            #in this case we are free to go ahead with the relaxation
            relaxation_flag = true
          else
            #if both conditions above do not hold we prevent relaxation from happening
            relaxation_flag = false
          end
        else
          #if we have to numerical values with different signs we are free to calculate a numerical root
          γ = Roots.find_zero(functional, (γlo, γhi), Roots.AlefeldPotraShi())
          #thus implying that we are free to go ahead with the relaxation procedure
          relaxation_flag = true
        end
        if γ < eps(typeof(γ))
          #if the relaxation parameter is smaller than machine precision we dont want to go ahead with the relaxation procedure as well
          relaxation_flag = false
        else
          #otherwise this is fine
          relaxation_flag = true
        end

        #with the proceudure above we determined if we can perform relaxation or not
        if relaxation_flag
          #first we view the step as accepted
          accept_step!(controller)
          integrator.naccept += 1

          @. unew = uold + γ * (unew - uold)
          copy!(uprev, unew)
          if tnew ≈ tend
            integrator.t = tnew
          else
            tγ = told + γ * (tnew - told)
            integrator.t = tγ
          end
          f!(fsalfirst, uprev, params, integrator.t)
          integrator.nf += 1
        else
          #if we were unable to perform the relaxation we retry with a smaller stepsize
          #since theory suggests that the relaxation problem as a unique solution for sufficient small step
          #in this case we choose half of the step size in the next try.
          reject_step!(controller)
          integrator.dt *= 0.5
          integrator.nreject += 1
        end
        if terminate_integration
          @warn "Terminating integration since no suitable relaxation parameter can be found"
          break
        end
      else
        integrator.t += integrator.dt
        copy!(uprev, u)
        copy!(fsalfirst, ktmp[end])
      end

      # callback
      callback.affect!(integrator)
    else
      # @warn "reject" integrator.t integrator.dt error_estimate dt_factor controller.err integrator.nreject integrator.naccept
      # error()
      reject_step!(controller)
      integrator.nreject += 1
    end
    #update the dt factor if step size control is active
    if adaptive && relaxation_flag
      integrator.dt *= dt_factor
    end

    if integrator.dt < 1.0e-14
      @error "time step too small" integrator.dt integrator.t integrator.naccept integrator.nreject
      error()
    end
  end

  return nothing
end

function integrate_naive(ode, alg, tol; relaxation = true, relaxation_at_last_step = true, adaptive = true, dt = 0.0, kwargs...)
  # Prepare callback
  saved_values = SavedValues(eltype(ode.tspan), NTuple{2, eltype(ode.u0)})
  callback = SavingCallback(ode.p.functionals, saved_values)

  tableau = ButcherTableau(alg)
  controller = default_controller(alg)
  sol = ode_solve(ode, tableau, solve_naive!;
                        dt = dt, abstol = tol, reltol = tol,
                        controller, relaxation, relaxation_at_last_step, adaptive, callback, kwargs...)
  return (; t = saved_values.t,
            entropy = first.(saved_values.saveval),
            error = last.(saved_values.saveval),
            nf = sol.destats.nf,
            naccept = sol.destats.naccept,
            nreject = sol.destats.nreject,
            sol,)
end


# FSAL-R combination of FSAL and relaxation approximating f(uⁿ⁺¹_γ)
@muladd function solve_fsalr!(integrator::Integrator; interpolate_FSAL = true)
  (; u, uprev, uembd, utmp, ktmp, fsalfirst,
     tableau, abstol, reltol, controller, relaxation, callback) = integrator
  (; A, b, bembd, c, order) = tableau
  f! = integrator.prob.f.f
  params = integrator.prob.p
  tend = last(integrator.prob.tspan)

  relaxation_at_last_step = integrator.relaxation_at_last_step
  adaptive = integrator.adaptive
  accept_flag = true
  relaxation_flag = true
  # solve until we have reached the final time
  while integrator.t < tend
    # adjust `dt` at the final time
    if integrator.t + integrator.dt > tend
      integrator.dt = tend - integrator.t
    end

    # compute explicit RK step with step size `dt`
    let i = 1 # first stage is special
      copy!(utmp[i], uprev)
      @. ktmp[i] = fsalfirst
    end
    for i in 2:length(b)
      copy!(utmp[i], uprev)
      for j in 1:i-1
        @. utmp[i] = utmp[i] + (A[i,j] * integrator.dt) * ktmp[j]
      end
      f!(ktmp[i], utmp[i], params, integrator.t + c[i] * integrator.dt)
      integrator.nf += 1
    end
    copy!(u,     uprev)
    copy!(uembd, uprev)
    for i in eachindex(b)
      @. u     = u     + (b[i]     * integrator.dt) * ktmp[i]
      @. uembd = uembd + (bembd[i] * integrator.dt) * ktmp[i]
    end
    if adaptive
      # compute error estimate
      error_estimate = compute_error_estimate(u, uprev, uembd, abstol, reltol)

      # adapt `dt`
      dt_factor = compute_dt_factor!(controller, error_estimate, order)

      accept_flag = accept_step(controller, dt_factor)
    end
    # accept or reject the step and update `dt`
    if accept_flag
      # @info "accept" integrator.t

      # We don't call `accept_step!(controller)` here
      # since we may reject the step later if relaxation fails.
      #check if we are at the last step and if we want to perform relaxation there
      if !relaxation_at_last_step && integrator.t + integrator.dt == tend
        relaxation = false
      end
      # perform relaxation
      if relaxation
        told = integrator.t
        uold = uprev
        tnew = integrator.t + integrator.dt
        unew = u
        temp = first(get_tmp_cache(integrator))
        entropy = params.entropy

        γ = one(tnew)
        terminate_integration = false
        γlo = (4//5) * one(γ)
        γhi = (6//5) * one(γ)
        functional_old = entropy(uold, params)

        functional = RelaxationFunctional(temp, unew, uold, params,
                                          params.entropy, functional_old)
        functional_lo = functional(γlo)
        functional_hi = functional(γhi)
        #first we check if we are able to perform relaxation with the values computed above
        if functional_lo * functional_hi > 0
          #in order to compute a root numerically we need different signs for functional_hi and functional_lo

          #If both values are close to machine precision anyway we make an exception from above rule,
          #since the relaxation parameter is defined as a root of a nonlinear equation
          if abs(functional_lo) + abs(functional_hi) < 10 * eps()
            #since surrounding values are close to machine precision,
            #we kind of have a free choice
            #in this case we choose γ = 1.0 since theory suggests that the relaxation parameter is close to 1 anyway
            #since the relaxation parameter is initiated with 1 we do not have to take any further actions

            #in this case we are free to go ahead with the relaxation
            relaxation_flag = true
          else
            #if both conditions above do not hold we prevent relaxation from happening
            relaxation_flag = false
          end
        else
          #if we have to numerical values with different signs we are free to calculate a numerical root
          γ = Roots.find_zero(functional, (γlo, γhi), Roots.AlefeldPotraShi())
          #thus implying that we are free to go ahead with the relaxation procedure
        end
        if γ < eps(typeof(γ))
           #if the relaxation parameter is smaller than machine precision we dont want to go ahead with the relaxation procedure as well
          relaxation_flag = false
        else
          #otherwise this is fine
          relaxation_flag = true
        end

        #with the proceudure above we determined if we can perform relaxation or not
        if relaxation_flag
          #if we use an adaptive stepsize control we first view the step as accepted
          integrator.naccept += 1
          if adaptive
            accept_step!(controller)
          end
          @. unew = uold + γ * (unew - uold)
          copy!(uprev, unew)
          if tnew ≈ tend
            integrator.t = tnew
          else
            tγ = told + γ * (tnew - told)
            integrator.t = tγ
          end
          # Approximate the RHS of the relaxed solution by an interpolation
          if interpolate_FSAL
            @. fsalfirst = ktmp[begin] + γ * (ktmp[end] - ktmp[begin])
          else
            @. fsalfirst = ktmp[end]
          end
          # f!(fsalfirst, uprev, params, integrator.t)
          # integrator.nf += 1
        else
          #if we were unable to perform the relaxation we retry with a smaller stepsize
          #since theory suggests that the relaxation problem as a unique solution for sufficient small step
          #in this case we choose half of the step size in the next try.
          integrator *= 0.5
          integrator.nreject += 1
        end
        if terminate_integration
          @warn "Terminating integration since no suitable relaxation parameter can be found"
          break
        end
      else
        integrator.t += integrator.dt
        copy!(uprev, u)
        copy!(fsalfirst, ktmp[end])
      end

      # callback
      callback.affect!(integrator)
    else
      # @warn "reject" integrator.t integrator.dt error_estimate dt_factor controller.err integrator.nreject integrator.naccept
      # error()
      reject_step!(controller)
      integrator.nreject += 1
    end
    if adaptive && relaxation_flag
      integrator.dt *= dt_factor
    end
    if integrator.dt < 1.0e-14
      @error "time step too small" integrator.dt integrator.t integrator.naccept integrator.nreject
      error()
    end
  end

  return nothing
end


function integrate_fsalr(ode, alg, tol; relaxation = true, relaxation_at_last_step = true, dt = 0.0, adaptive = true, kwargs...)
  if !relaxation
    @warn "Running FSAL-R without relaxation"
  end

  # Prepare callback
  saved_values = SavedValues(eltype(ode.tspan), NTuple{2, eltype(ode.u0)})
  callback = SavingCallback(ode.p.functionals, saved_values)
  tableau = ButcherTableau(alg)
  controller = default_controller(alg)
  sol = ode_solve(ode, tableau, solve_fsalr!;
                        dt = dt, abstol = tol, reltol = tol,
                        controller, relaxation, relaxation_at_last_step, adaptive, callback, kwargs...)
  return (; t = saved_values.t,
            entropy = first.(saved_values.saveval),
            error = last.(saved_values.saveval),
            nf = sol.destats.nf,
            naccept = sol.destats.naccept,
            nreject = sol.destats.nreject,
            sol,)
end


# FSAL-R combination of FSAL and relaxation approximating f(uⁿ⁺¹)
@muladd function solve_rfsal!(integrator::Integrator;
                              interpolate_FSAL = true,
                              error_estimate_relax_main = true,
                              error_estimate_relax_embedded = true)
  (; u, uprev, uembd, utmp, ktmp, fsalfirst,
     tableau, abstol, reltol, controller, relaxation, callback) = integrator
  (; A, b, bembd, c, order) = tableau

  relaxation_at_last_step = integrator.relaxation_at_last_step

  f! = integrator.prob.f.f
  params = integrator.prob.p
  tend = last(integrator.prob.tspan)
  fsallast = last(get_tmp_cache(integrator))

  dt_factor = one(integrator.dt)

  if relaxation
    reactivate_relaxation_flag = true
  else
    reactivate_relaxation_flag = false
  end

  relaxation_flag = true
  accept_flag = true
  dt_factor = one(integrator.t)
  # solve until we have reached the final time
  while integrator.t < tend
    # adjust `dt` at the final time
    if integrator.t + integrator.dt > tend
      integrator.dt = tend - integrator.t
    end

    # compute explicit RK step with step size `dt`
    let i = 1 # first stage is special
      copy!(utmp[i], uprev)
      @. ktmp[i] = fsalfirst
    end
    # do not compute the last stage == FSAL stage
    for i in 2:(length(b)-1)
      copy!(utmp[i], uprev)
      for j in 1:i-1
        @. utmp[i] = utmp[i] + (A[i,j] * integrator.dt) * ktmp[j]
      end
      f!(ktmp[i], utmp[i], params, integrator.t + c[i] * integrator.dt)
      integrator.nf += 1
    end
    copy!(u, uprev)
    for i in 1:(length(b)-1)
      @. u = u + (b[i] * integrator.dt) * ktmp[i]
    end

    if !error_estimate_relax_main
      utmp[1] .= u
    end
    # perform relaxation
    tnew = integrator.t + integrator.dt
    γ = one(tnew)

    #check if we are at the last step and if we want to perform relaxation there
    if !relaxation_at_last_step && integrator.t + integrator.dt == tend
      relaxation = false
    end
    if relaxation
      told = integrator.t
      uold = uprev
      unew = u
      temp = first(get_tmp_cache(integrator))
      terminate_integration = false
      entropy = params.entropy



      γlo = (4//5) * one(γ)
      γhi = (6//5) * one(γ)
      functional_old = entropy(uold, params)

      functional = RelaxationFunctional(temp, unew, uold, params,
                                        params.entropy, functional_old)
      functional_lo = functional(γlo)
      functional_hi = functional(γhi)

      #first we check if we are able to perform relaxation with the values computed above
      if functional_lo * functional_hi > 0
        #in order to compute a root numerically we need different signs for functional_hi and functional_lo

        #If both values are close to machine precision anyway we make an exception from above rule,
        #since the relaxation parameter is defined as a root of a nonlinear equation
        if abs(functional_lo) + abs(functional_hi) < 10 * eps()
          #since surrounding values are close to machine precision,
          #we kind of have a free choice
          #in this case we choose γ = 1.0 since theory suggests that the relaxation parameter is close to 1 anyway
          #since the relaxation parameter is initiated with 1 we do not have to take any further actions

          #in this case we are free to go ahead with the relaxation
          relaxation_flag = true
        else
          #if both conditions above do not hold we prevent relaxation from happening
          relaxation_flag = false
        end
      else
        #if we have to numerical values with different signs we are free to calculate a numerical root
        γ = Roots.find_zero(functional, (γlo, γhi), Roots.AlefeldPotraShi())
        #thus implying that we are free to go ahead with the relaxation procedure
        relaxation_flag = true
      end
      if γ < eps(typeof(γ))
        #if the relaxation parameter is smaller than machine precision we dont want to go ahead with the relaxation procedure as well
        relaxation_flag = false
      else
        #otherwise this is fine
        relaxation_flag = true
      end

      #with the proceudure above we determined if we can perform relaxation or not
      if relaxation_flag
        #in this case we are free to perform relaxation
        #to continue the procedure we need to check if the FSAL stepsize-control accepts the step as well for this we need
        accept_flag = true

        @. unew = uold + γ * (unew - uold)
        if !(tnew ≈ tend)
          tnew = told + γ * (tnew - told)
        end

        # Compute the RHS of the relaxed solution
        f!(fsallast, unew, params, tnew)
        integrator.nf += 1

        # optionally approximate f(unew) for the embedded method
        if interpolate_FSAL
          inv_γ = inv(γ)
          @. ktmp[end] = fsalfirst + inv_γ * (fsallast - fsalfirst)
        else
          @. ktmp[end] = fsallast
        end
      else
        #if we were unable to perform the relaxation we retry with a smaller stepsize
        #since theory suggests that the relaxation problem as a unique solution for sufficient small step
        #we reject the step
        accept_flag = false
      end
    else
      # compute f(unew) for the embedded method
      f!(fsallast, u, params, integrator.t + integrator.dt)
      copy!(ktmp[end], fsallast)
      integrator.nf += 1
    end

    # compute embedded solution using an approximation of f(unew) when
    # relaxation is used

    if accept_flag
      copy!(uembd, uprev)
      if error_estimate_relax_embedded
        for i in eachindex(b)
          @. uembd = uembd + (γ * bembd[i] * integrator.dt) * ktmp[i]
        end
      else
        #in this case we want to use the unrelaxed embedded solution
        for i in eachindex(b)
          @. uembd = uembd + (bembd[i] * integrator.dt) * ktmp[i]
        end
      end
      # compute error estimate
      if error_estimate_relax_main
        error_estimate = compute_error_estimate(u, uprev, uembd, abstol, reltol)
      else
        error_estimate = compute_error_estimate(utmp[1], uprev, uembd, abstol, reltol)
      end
      # adapt `dt`
      dt_factor = compute_dt_factor!(controller, error_estimate, order)
      accept_flag = accept_step(controller, dt_factor)
      #here we check if the FSAL-stepsize control accepts the step as well
    end
    # accept or reject the step and update `dt`
    if accept_flag
      # @info "accept" integrator.t
      accept_step!(controller)
      integrator.naccept += 1

      integrator.t = tnew
      copy!(uprev, u)
      copy!(fsalfirst, fsallast)

      # callback
      callback.affect!(integrator)
    else
      # @warn "reject" integrator.t integrator.dt error_estimate dt_factor controller.err integrator.nreject integrator.naccept
      # error()
      reject_step!(controller)
      integrator.nreject += 1
    end

    if relaxation_flag
      #if relaxation was successful we adjust the stepsize according to the FSAL-stepsize control
      integrator.dt *= dt_factor
    else
      #if relaxation was unsuccessful we adjust the stepsize according to our relaxational stepsize control
      integrator.dt *= 0.5
    end
    if !relaxation && reactivate_relaxation_flag
      #this ensures that relaxation gets reactivated if the last step gets rejected
      #and we have to retry with a smaller step_size
      #if the last step gets accepted we are done anyway and this operation does not change anything
      relaxation = true
    end
    if integrator.dt < 1.0e-14
      @error "time step too small" integrator.dt integrator.t integrator.naccept integrator.nreject
      @error stacktrace(catch_backtrace())
      error()
    end
  end

  return nothing
end

function integrate_rfsal(ode, alg, tol; relaxation = true, relaxation_at_last_step = true, adaptive = true, kwargs...)
  if !relaxation
    @warn "Running R-FSAL without relaxation"
  end

  # Prepare callback
  saved_values = SavedValues(eltype(ode.tspan), NTuple{2, eltype(ode.u0)})
  callback = SavingCallback(ode.p.functionals, saved_values)

  tableau = ButcherTableau(alg)
  controller = default_controller(alg)
  sol = ode_solve(ode, tableau, solve_rfsal!;
                        dt = 0.0, abstol = tol, reltol = tol,
                        controller, relaxation, relaxation_at_last_step, adaptive, callback, kwargs...)
  return (; t = saved_values.t,
            entropy = first.(saved_values.saveval),
            error = last.(saved_values.saveval),
            nf = sol.destats.nf,
            naccept = sol.destats.naccept,
            nreject = sol.destats.nreject,
            sol,)
end

# Compute a work precision diagram
function work_precision_diagram(ode, alg, tolerances; legend = true, legend_on_plot = true, kwargs...)

  ncol_counter = 0

  fig = PyPlot.figure()
  fig_legend = PyPlot.figure()

  ax = fig.add_subplot(1,1,1)
  ax.set_yscale("log")
  ax.set_xscale("log")

  ax.set_xlabel("# RHS Calls")
  ax.set_ylabel("Error")

  ncalls = Vector{Int}()
  errors = Vector{Float64}()

  empty!(ncalls)
  empty!(errors)
  for tol in tolerances
      res = integrate_naive(ode, alg, tol; relaxation = false, relaxation_at_last_step = false, kwargs...)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
  end
  ax.scatter(ncalls, errors, label = "Baseline", marker = :x)
  ncol_counter += 1

  # naive relaxation, manual implementation
  empty!(ncalls)
  empty!(errors)
  for tol in tolerances
      res = integrate_naive(ode, alg, tol; relaxation = true, relaxation_at_last_step = false, kwargs...)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
  end
  ax.scatter(ncalls, errors, label = "Naive", marker = :x)
  ncol_counter += 1

  # FSAL-R
  empty!(ncalls)
  empty!(errors)
  for tol in tolerances
      res = integrate_fsalr(ode, alg, tol; relaxation = true, interpolate_FSAL = true, relaxation_at_last_step = false, kwargs...)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
  end
  ax.scatter(ncalls, errors, label = "FSAL-R", marker = "X")
  ncol_counter += 1
  # R-FSAL
  empty!(ncalls)
  empty!(errors)
  for tol in tolerances
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = true, interpolate_FSAL = true, relaxation = true, error_estimate_relax_main = true, relaxation_at_last_step = false, kwargs...)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
  end

  ax.scatter(ncalls, errors, label = "R-FSAL", marker = "*")
  ncol_counter += 1

  if legend
    if legend_on_plot
      ax.legend(loc = "center left", bbox_to_anchor = (1.04, 0.5))
    else
      fig_legend.legend(ax.get_legend_handles_labels()..., ncol = Int(floor(ncol_counter )))
    end
  end

  fig.tight_layout()

  return fig, fig_legend
end

#Compute a work precision diagram with the RFSAL-method
function work_precision_diagram_rfsal(ode , alg, tolerances; legend = true, legend_on_plot = true)
  #keyword FSAL_interpolate determines whether the naive or the interpolated approximation of the last stage is used
  #keyword error_estimate_relax_embedded determines whether the time step \Delta t or \gamma \Delta t is used for the embedded method.
  #keyword error_estimate_relax_main determines whether we use u^{n + 1} or u^{n + 1}_{\gamma} for error estimation.
  ncol_counter = 0

  fig = PyPlot.figure()
  fig_legend = PyPlot.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_yscale("log")
  ax.set_xscale("log")

  ax.set_xlabel("# RHS Calls")
  ax.set_ylabel("Error")

  ncalls = Vector{Int}()
  errors = Vector{Float64}()

  empty!(ncalls)
  empty!(errors)
  #\Delta t_{\gamma}, naive, u^{n + 1}

  for tol in tolerances
    try
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = true, interpolate_FSAL = false, relaxation = true, error_estimate_relax_main = false, relaxation_at_last_step = false)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
    catch
      @warn L"\Delta t_{\gamma}, naive, unrelaxed, u^{n + 1} failed for testproblem" tol
    end
  end
  s_1 = L"\Delta t_{\gamma}"
  s_2 = ", simple, unrelaxed, "
  s_3 = L"u^{n + 1}"
  label_string = string(s_1, s_2, s_3)
  ax.scatter(ncalls, errors, label = label_string, marker = "s")
  ncol_counter += 1

  empty!(ncalls)
  empty!(errors)
  #\Delta t_{\gamma}, naive, u^{n + 1}_{\gamma}

  for tol in tolerances
    try
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = true, interpolate_FSAL = false, relaxation = true, error_estimate_relax_main = true, relaxation_at_last_step = false)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
    catch
      @warn L"\Delta t_{\gamma}, naive, relaxed, u^{n + 1}_{\gamma} failed for testproblem" tol
    end
  end

  s_1 = L"\Delta t_{\gamma}"
  s_2 = ", simple, relaxed, "
  s_3 = L" u^{n + 1}_{\gamma}"
  label_string = string(s_1, s_2, s_3)
  ax.scatter(ncalls, errors, label = label_string, marker = "p")
  ncol_counter += 1

  empty!(ncalls)
  empty!(errors)
  #\Delta t_{\gamma}, interpolated, u^{n + 1}

  for tol in tolerances
    try
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = true, interpolate_FSAL = true, relaxation = true, error_estimate_relax_main = false, relaxation_at_last_step = false)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
    catch
      @warn L"\Delta t_{\gamma}, interpolated, unrelaxed, u^{n + 1} failed for testproblem"
    end
  end

  s_1 = L"\Delta t_{\gamma}"
  s_2 = ", interpolated, unrelaxed, "
  s_3 = L" u^{n + 1}"
  label_string = string(s_1, s_2, s_3)
  ax.scatter(ncalls, errors, label = label_string, marker = "D")
  ncol_counter += 1

  empty!(ncalls)
  empty!(errors)
  #\Delta t_{\gamma}, interpolated, u^{n + 1}_{\gamma}

    for tol in tolerances
      try
        res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = true, interpolate_FSAL = true, relaxation = true, error_estimate_relax_main = true, relaxation_at_last_step = false)
        push!(ncalls, res.nf)
        push!(errors, last(res.error))
      catch
        @warn L"\Delta t_{\gamma}, interpolated, relaxed, u^{n + 1}_{\gamma} failed for testproblem" tol
      end
    end

  s_1 = L"\Delta t_{\gamma}"
  s_2 = ", interpolated, relaxed, "
  s_3 = L" u^{n + 1}_{\gamma}"
  label_string = string(s_1, s_2, s_3)

  ax.scatter(ncalls, errors, label = label_string, marker = "X")
  ncol_counter += 1

  if legend
    if legend_on_plot
      ax.legend(loc = "center left", bbox_to_anchor = (1.04, 0.5))
    else
      fig_legend.legend(ax.get_legend_handles_labels()..., ncol = Int(floor(ncol_counter / 2)))
    end
  end

  fig.tight_layout()

  return fig, fig_legend
end

function work_precision_diagram_rfsal_bad_variants(legend = true)
  #this function gathers all the bad options for the R-FSAL variant
  #use at own risk.

  #keyword FSAL_interpolate determines whether the naive or the interpolated approximation of the last stage is used
  #keyword error_estimate_relax_embedded determines whether the time step \Delta t or \gamma \Delta t is used for the embedded method.
  #keyword error_estimate_relax_main determines whether we use u^{n + 1} or u^{n + 1}_{\gamma} for error estimation.

  fig = PyPlot.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_yscale("log")
  ax.set_xscale("log")

  ax.set_xlabel("# RHS Calls")
  ax.set_ylabel("Error")

  ncalls = Vector{Int}()
  errors = Vector{Float64}()
  empty!(ncalls)
  empty!(errors)
  #\Delta t, naive, u^{n + 1}

  for tol in tolerances
    try
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = false, interpolate_FSAL = false, relaxation = true, error_estimate_relax_main = false)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))

    catch
      @warn L"\Delta t, naive, u^{n + 1} failed for testproblem" tol
    end
  end
  ax.scatter(ncalls, errors, label = L"\Delta t, naive, unrelaxed, u^{n + 1}", marker = "+")

  empty!(ncalls)
  empty!(errors)
  #\Delta t, naive, u^{n + 1}_{\gamma}
  for tol in tolerances
    try
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = false, interpolate_FSAL = false, relaxation = true, error_estimate_relax_main = true)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
    catch
      @warn L"\Delta t, naive, relaxed, u^{n + 1}_{\gamma} failed for testproblem" tol
    end
  end
  ax.scatter(ncalls, errors, label = L"\Delta t, naive, relaxed, u^{n + 1}_{\gamma}", marker = "X")
  empty!(ncalls)
  empty!(errors)
  #\Delta t, interpolated, u^{n + 1}

  for tol in tolerances
    try
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = false, interpolate_FSAL = true, relaxation = true, error_estimate_relax_main = false)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
    catch
        @warn L"\Delta t, interpolated, unrelaxed, u^{n + 1} failed for testproblem" tol
    end
  end
  ax.scatter(fig, ncalls, errors, label = L"\Delta t, interpolated, unrelaxed, u^{n + 1}", marker = "*")

  empty!(ncalls)
  empty!(errors)
  #\Delta t, interpolated, u^{n + 1}_{\gamma}

  for tol in tolerances
    try
      res = integrate_rfsal(ode, alg, tol; error_estimate_relax_embedded = false, interpolate_FSAL = true, relaxation = true, error_estimate_relax_main = true)
      push!(ncalls, res.nf)
      push!(errors, last(res.error))
    catch
      @warn L"\Delta t, interpolated, relaxed, u^{n + 1}_{\gamma} failed for testproblem" tol
    end
  end

  ax.scatter(ncalls, errors, label = L"\Delta t, interpolated, relaxed, u^{n + 1}_{\gamma}", marker = "h")

  if legend
    fig_legend.legend(ax.get_legend_handles_labels())
  end

  return fig, fig_legend
end

function work_precision_diagram_fsalr(ode, alg, tolerances; legend = true, legend_on_plot = true, set_xticks = false)

  ncol_counter = 0

  fig = PyPlot.figure()
  fig_legend = PyPlot.figure()
  ax = fig.add_subplot(1,1,1)
  ax.loglog()
  ax.set_xlabel("# RHS Calls")
  ax.set_ylabel("Error")
  #ax.set_xticks([10^2, 2*10^2, 3*10^2, 6 * 10^2])
  ncalls = Vector{Int}()
  errors = Vector{Float64}()
  empty!(ncalls)
  empty!(errors)

  # using the naive approach as a first stage approximation

  for tol in tolerances
    res = integrate_fsalr(ode, alg, tol; relaxation = true, interpolate_FSAL = false, relaxation_at_last_step = false)
    push!(ncalls, res.nf)
    push!(errors, last(res.error))
  end
  ax.scatter(ncalls, errors, label = "simple", marker = "*")
  ncol_counter += 1


  #ax.set_xticks([])
  empty!(ncalls)
  empty!(errors)

  # using the interpolation formula as a first stage approximation
  for tol in tolerances
    res = integrate_fsalr(ode, alg, tol; relaxation = true, interpolate_FSAL = true, relaxation_at_last_step = false)
    push!(ncalls, res.nf)
    push!(errors, last(res.error))
  end
  ax.scatter(ncalls, errors, label = "interpolation", marker = ".")
  ncol_counter += 1
  if set_xticks
    ticks_vec = collect(LinRange(ncalls[1], ncalls[end], 4))
    ax.set_xticks(ticks_vec)
    ax.get_xaxis().set_major_formatter(PyPlot.matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(PyPlot.matplotlib.ticker.NullFormatter())
  end

  if legend
    if legend_on_plot
      ax.legend(loc = "center left", bbox_to_anchor = (1.04, 0.5))
    else
      fig_legend.legend(ax.get_legend_handles_labels()..., ncol = ncol_counter)
    end
  end
  #PyPlot.minorticks_off()
  fig.tight_layout()
  return fig, fig_legend
end

function convergence_diagram_fsalr(ode, alg, step_sizes, order; legend = true)
  fig = PyPlot.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_yscale("log")
  ax.set_xscale("log")

  ax.set_xlabel(L"h")
  ax.set_ylabel("Error")

  errors = Vector{Float64}()

  empty!(errors)
  for step in step_sizes
    res = integrate_fsalr(ode, alg, 0.0; relaxation = true, interpolate_FSAL = true, relaxation_at_last_step = false, dt = step, adaptive = false)
    push!(errors, last(res.error))
  end
  ax.plot(step_sizes, errors, label = "interpolation")


  empty!(errors)
  for step in step_sizes
    res = integrate_fsalr(ode, alg, 0.0; relaxation = true, interpolate_FSAL = false, relaxation_at_last_step = false, dt = step, adaptive = false)
    push!(errors, last(res.error))
  end
  ax.plot(step_sizes, errors, label = "naive")


  #add order convergence line
  #we add some offset to the data we are looking at, in order to distinguish between the data and the convergence line
  h_0 = minimum(step_sizes)
  e_0 = minimum(errors) * 1e1
  h_1 = maximum(step_sizes)
  e_1 = e_0 * (h_1 / h_0)^order
  e_1_order_plus = e_0 *  (h_1 / h_0)^(order + 1)
  e_1_order_minus = e_0 *  (h_1 / h_0)^(order - 1)

  ax.plot([h_0, h_1], [e_0, e_1], marker = ".", label = L"\mathcal{O}(\Delta t^%$order)")
  ax.plot([h_0, h_1], [e_0, e_1_order_plus], marker = ".", label = L"\mathcal{O}(\Delta t^%$(order + 1) )")
  ax.plot([h_0, h_1], [e_0, e_1_order_minus], marker = ".", label = L"\mathcal{O}(\Delta t^%$(order - 1))")

  if legend
    ax.legend()
  end

  return fig
end

function convergence_diagram_naive(ode, alg, step_sizes, order)
  fig = PyPlot.figure()
  ax = fig.add_subplot(1,1,1)
  ax.set_yscale("log")
  ax.set_xscale("log")

  ax.set_xlabel(L"h")
  ax.set_ylabel("Error")

  errors = Vector{Float64}()

  empty!(errors)

  for step in step_sizes
    res = integrate_naive(ode, alg, 0.0; relaxation = false, adaptive = false, dt = step, relaxation_at_last_step = false)
    push!(errors, last(res.error))
  end
  ax.plot(step_sizes, errors, label = "without relaxation")

  empty!(errors)
  for step in step_sizes
    res = integrate_naive(ode, alg, 0.0; relaxation = true, adaptive = false, dt = step, relaxation_at_last_step = false)
    push!(errors, last(res.error))
  end
  ax.plot(step_sizes, errors, label = "naive")

  #add order convergence line
  #we add some offset to the data we are looking at, in order to distinguish between the data and the convergence line
  h_0 = minimum(step_sizes)
  e_0 = minimum(errors) * 1e1
  h_1 = maximum(step_sizes)
  e_1 = e_0 * (h_1 / h_0)^order
  e_1_order_plus = e_0 *  (h_1 / h_0)^(order + 1)
  e_1_order_minus = e_0 *  (h_1 / h_0)^(order - 1)

  ax.plot([h_0, h_1], [e_0, e_1], marker =:circle, label = L"\mathcal{O}(\Delta t^%$order)")
  ax.plot([h_0, h_1], [e_0, e_1_order_plus], marker =:circle, label = L"\mathcal{O}(\Delta t^%$(order + 1) )")
  ax.plot([h_0, h_1], [e_0, e_1_order_minus], marker =:circle, label = L"\mathcal{O}(\Delta t^%$(order - 1))")

  return fig
end

function generate_error_fsalr(ode, alg, step_sizes)
  errors_interpolation = Vector{Float64}()
  errors_naive = Vector{Float64}()

  empty!(errors_interpolation)
  empty!(errors_naive)

  for step in step_sizes
    res = integrate_fsalr(ode, alg, 0.0; relaxation = true, interpolate_FSAL = true, relaxation_at_last_step = false, dt = step, adaptive = false)
    push!(errors_interpolation, last(res.error))
  end
  for step in step_sizes
    res = integrate_fsalr(ode, alg, 0.0; relaxation = true, interpolate_FSAL = false, relaxation_at_last_step = false, dt = step, adaptive = false)
    push!(errors_naive, last(res.error))
  end

  return (errors_interpolation, errors_naive)
end

function add_plot_fsalr_convergence(fig, step_sizes, errors_interpolation, errors_naive)

  plot!(fig, step_sizes, errors_interpolation, label = "interpolation")
  plot!(fig, step_sizes, errors_naive, label = "naive")

  return nothing
end

function generate_plot_fsalr_convergence(ode, step_sizes; legend = true, legend_on_plot = false)
  ncol_counter = 0

  fig = PyPlot.figure()
  fig_legend = PyPlot.figure()

  ax = fig.add_subplot(1,1,1)
  ax.set_yscale("log")
  ax.set_xscale("log")

  ax.set_xlabel(L"\Delta t")
  ax.set_ylabel("Error")

  errors_interpolation = Vector{Float64}()
  errors_naive = Vector{Float64}()

  empty!(errors_interpolation)
  empty!(errors_naive)

  alg = BS3()
  errors_interpolation, errors_naive = generate_error_fsalr(ode, alg, step_sizes)

  ax.plot(step_sizes, errors_interpolation; label = "BS3 interpolation", linestyle = "dotted", color =:lightgreen)
  ncol_counter += 1
  ax.plot(step_sizes, errors_naive; label = "BS3 naive", linestyle = "dashed", color =:green)
  ncol_counter += 1

  #add order curves
  order = 3
  h_0 = step_sizes[1]
  e_0 = minimum(errors_interpolation)
  h_1 = step_sizes[Int(length(step_sizes) / 2)]
  e_1 = e_0 * (h_1 / h_0)^order
  ax.plot([h_0, h_1], [e_0, e_1];
        marker = ".", color =:grey, label = L"\mathcal{O}(\Delta t^%$order)")
  ncol_counter += 1
  empty!(errors_interpolation)
  empty!(errors_naive)

  alg = DP5()
  errors_interpolation, errors_naive = generate_error_fsalr(ode, alg, step_sizes)

  ax.plot(step_sizes, errors_interpolation; label = "DP5 interpolation", linestyle = "dotted", color =:blue)
  ncol_counter += 1
  ax.plot(step_sizes, errors_naive; label = "DP5 naive", linestyle ="solid", color =:orange)
  ncol_counter += 1

  #add order curves
  order = 5
  h_0 = step_sizes[1]
  e_0 = minimum(errors_interpolation)
  h_1 = step_sizes[Int(length(step_sizes) / 2)]
  e_1 = e_0 * (h_1 / h_0)^order
  ax.plot([h_0, h_1], [e_0, e_1];marker = "X", color =:grey, label = L"\mathcal{O}(\Delta t^%$order)")
  ncol_counter += 1

  empty!(errors_interpolation)
  empty!(errors_naive)

  alg = RK4()
  errors_interpolation, errors_naive = generate_error_fsalr(ode, alg, step_sizes)

  ax.plot(step_sizes, errors_interpolation; label = "RK4 interpolation", linestyle ="dashdot", color =:red)
  ncol_counter += 1
  ax.plot(step_sizes, errors_naive; label = "RK4 naive", linestyle="-.", color=:cyan)
  ncol_counter += 1

  #add order curves
  order = 4
  h_0 = step_sizes[1]
  e_0 = minimum(errors_interpolation)
  h_1 = step_sizes[Int(length(step_sizes) / 2)]
  e_1 = e_0 * (h_1 / h_0)^order
  ax.plot([h_0, h_1], [e_0, e_1]; marker ="s", color =:grey, label = L"\mathcal{O}(\Delta t^%$order)")
  ncol_counter += 1

  #adding a legend
  if legend
    if legend_on_plot
      ax.legend(loc = "center left", bbox_to_anchor = (1.04, 0.5))
    else
      fig_legend.legend(ax.get_legend_handles_labels()..., ncol = Int(floor(ncol_counter / 3)))
    end

  end
  return fig, fig_legend
end


# Main functions
function plots_rfsal_section()
  figdir = joinpath(@__DIR__, "Plots_RFSAL")
  isdir(figdir) || mkdir(figdir)

  #harmonic oscillator
  # BS3
  alg = BS3()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = harmonic_osc_setup()
  fig, fig_legend = work_precision_diagram_rfsal(ode, alg, tolerances; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_harmonic_osc__work_precision_rfsal.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_harmonic_osc__work_precision_rfsal_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")
  #DP5
  alg = DP5()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = harmonic_osc_setup()
  fig, fig_legend = work_precision_diagram_rfsal(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_harmonic_osc__work_precision_rfsal.pdf"), bbox_inches = "tight")
  PyPlot.close("all")
  #conserved exponential entropy
  #BS3
  alg = BS3()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = conserved_exponential_energy_setup()
  fig, fig_legend = work_precision_diagram_rfsal(ode, alg, tolerances; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_conserved_exponential_energy__work_precision_rfsal.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_conserved_exponential_energy__work_precision_rfsal_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")
  #DP5
  alg = DP5()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = conserved_exponential_energy_setup()
  fig, fig_legend = work_precision_diagram_rfsal(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_conserved_exponential_energy__work_precision_rfsal.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  @info "Results saved in directory `figdir`" figdir

  return nothing
end

function plots_fsalr_section()
  figdir = joinpath(@__DIR__, "Plots_FSALR")
  isdir(figdir) || mkdir(figdir)
  PyPlot.ioff()
  ##convergence plots
  #harmonic oscillator problem
  ode = harmonic_osc_setup()
  step_sizes = 10.0 .^ range(-2.5, 0.0, length = 100)

  fig, fig_legend = generate_plot_fsalr_convergence(ode, step_sizes; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__harmonic_osc__fsalr_convergence.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__harmonic_osc__fsalr_convergence_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #nonlinear oscillator
  ode = nonlinear_osc_setup()
  step_sizes = 10.0 .^ range(-2.5, 0.0, length = 100)

  fig, fig_legend = generate_plot_fsalr_convergence(ode, step_sizes; legend = false, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__nonlinear_osc__fsalr_convergence.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #harmonic oscillator problem time dependent bounded
  ode = harmonic_osc_time_dependent_bounded_setup()
  step_sizes = 10.0 .^ range(-2.5, 0.0, length = 100)

  fig, fig_legend = generate_plot_fsalr_convergence(ode, step_sizes; legend = false)
  fig.savefig(joinpath(figdir, "Result__harmonic_osc_time_dependent_bounded__fsalr_convergence.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #conserved exponential entropy
  ode = conserved_exponential_energy_setup()
  step_sizes = 10.0 .^ range(-2.5, 0.0, length = 100)

  fig, fig_legend = generate_plot_fsalr_convergence(ode, step_sizes; legend = false)
  fig.savefig(joinpath(figdir, "Result__conserved_exponential_entropy__fsalr_convergence.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  ##comparison plots
  #nonlinear pendulum
  alg = BS3()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = nonlinear_pendulum_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_nonlinear_pendulum__work_precision_fsalr.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_nonlinear_pendulum__work_precision_fsalr_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = nonlinear_pendulum_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = false, set_xticks = true)
  fig.savefig(joinpath(figdir, "Result__DP5_nonlinear_pendulum__fsalr_comparison.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #conserverd exponential entropy
  alg = BS3()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = conserved_exponential_energy_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_conserved_exponential_entropy__fsalr_comparison.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_conserved_exponential_entropy__fsalr_comparison_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = conserved_exponential_energy_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_conserved_exponential_entropy__fsalr_comparison.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #harmonic oscillator problem
  alg = BS3()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = harmonic_osc_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_harmonic_osc__fsalr_comparison.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_harmonic_osc__fsalr_comparison_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = harmonic_osc_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_harmonic_osc__fsalr_comparison.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #harmonic oscillator problem time dependent bounded
  alg = BS3()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = harmonic_osc_time_dependent_bounded_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_harmonic_osc_time_dependent_bounded__fsalr_comparison.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_harmonic_osc_time_dependent_bounded__fsalr_comparison_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = harmonic_osc_time_dependent_bounded_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_harmonic_osc_time_dependent_bounded__fsalr_comparison.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #nonlinear oscillator
  alg = BS3()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = nonlinear_osc_setup()

  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = true, legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_nonlinear_osc__fsalr_comparison.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_nonlinear_osc__fsalr_comparison_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5()
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = nonlinear_osc_setup()
  fig, fig_legend = work_precision_diagram_fsalr(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_nonlinear_osc__fsalr_comparison.pdf"), bbox_inches = "tight")
  PyPlot.close("all")


  @info "Results saved in directory `figdir`" figdir
end

function plots_main()
  figdir = joinpath(@__DIR__, "Plots_main")
  isdir(figdir) || mkdir(figdir)

  PyPlot.ioff()
  #BBM
  alg = BS3(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = bbm_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_bbm__work_precision.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_bbm__work_precision_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = bbm_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_bbm__work_precision.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #linear advection
  alg = BS3(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = linear_transport_DG_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_linear_advection__work_precision.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = RDPK3SpFSAL49(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10]
  ode = linear_transport_DG_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__RDPK3SpFSAL49_linear_advection__work_precision.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #conserved exponential entropy
  alg = BS3(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = conserved_exponential_energy_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_conserved_exponential_energy__work_precision.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_conserved_exponential_energy__work_precision_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = conserved_exponential_energy_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_conserved_exponential_energy__work_precision.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  #nonlinear oscillator
  alg = BS3(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = nonlinear_osc_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend_on_plot = false)
  fig.savefig(joinpath(figdir, "Result__BS3_nonlinear_osc__work_precision.pdf"), bbox_inches = "tight")
  fig_legend.savefig(joinpath(figdir, "Result__BS3_nonlinear_osc__work_precision_legend.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  alg = DP5(step_limiter! = compute_relaxation_parameter!)
  tolerances = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
  ode = nonlinear_osc_setup()
  fig, fig_legend = work_precision_diagram(ode, alg, tolerances; legend = false)
  fig.savefig(joinpath(figdir, "Result__DP5_nonlinear_osc__work_precision.pdf"), bbox_inches = "tight")
  PyPlot.close("all")

  @info "Results saved in directory `figdir`" figdir

  return nothing
end

