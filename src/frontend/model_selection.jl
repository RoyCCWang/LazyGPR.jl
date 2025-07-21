# depends on types defined in solvers.jl

"""
    struct OptimContainer{T}
        lbs::Vector{T}
        ubs::Vector{T}
        p0s::Vector{Vector{T}}
    end
A container for optimization problems.
`lbs` and `ubs` are the lower and upper bounds, respectively.
`p0s` is an array of initial guesses.
"""
struct OptimContainer{T}
    lbs::Vector{T}
    ubs::Vector{T}
    p0s::Vector{Vector{T}}
end

"""
    optimize_kernel_hp(
        alg_trait::ExtensionPkgs,
        kernel::PositiveDefiniteKernel,
        ms_trait::ModelSelection,
        model::Union{LazyGP,GPData},
        config::HyperparameterInferenceInfo,
        solver_config::SolverConfig,
        optim_info::OptimContainer{T},
    ) where T <: Real

Fits the GPR hyperparameters for a single-process setup.
"""
function optimize_kernel_hp(
        alg_trait::ExtensionPkgs,
        kernel::PositiveDefiniteKernel,
        ms_trait::ModelSelection,
        model::Union{LazyGP, GPData},
        config::HyperparameterInferenceInfo,
        solver_config::SolverConfig,
        optim_info::OptimContainer{T},
    ) where {T <: Real}

    costfunc, Bs_info = create_hp_cost(
        resolvetrait(config),
        ms_trait,
        model,
        kernel,
        config,
    )

    # optimize.
    dek_vars, dek_star = run_optimizer(
        alg_trait,
        costfunc, optim_info.lbs, optim_info.ubs, solver_config, optim_info.p0s,
    )

    # clean up.
    free_hp_cost(Bs_info)

    return dek_vars, dek_star
end

"""
    @kwdef struct KernelOptimConfig{T}
        a_lb::T
        a_ub::T
        κ_ub::T
        N_initials_a::Int = 100
        N_initials_κ::Int = 100
        width_factor::T = one(T)
        height_factor::T = convert(T, 1/2)
    end

This configuration is used for the two-stage approach to optimizing DE kernels.
    
In the first stage, the stationary kernel is optimized.

In the second stage, a canonical kernel is formed based on the solution of the stationary kernel's hyperparameters, and the gain parameter is fitted with the canonical kernel bandwidth being fixed.

About some of the field members:
- `N_initials_a` and `N_initials_κ` specifies how many initial guesses are used in the optimization algorithm, sampled at uniformly spaced intervals from the lowerbound to the upperbounds.

- The gain parameter is `κ`.

- The bandwidth parameter is `a` for the `SqExpKernel` and `WendlandSpline` kernels.

- `width_factor` and  `height_factor` contols how much larger or smaller the DE kernel should be with respect to the bandwidth parameter `a` from the stationary kernel's optimized solution.

- The default setting is for the case when the DE kernel should use the stationary kernel's solution as its canonical kernel.
"""
@kwdef struct KernelOptimConfig{T}
    a_lb::T
    a_ub::T
    κ_ub::T
    N_initials_a::Int = 100
    N_initials_κ::Int = 100
    width_factor::T = one(T) # default is to not change the bandwidth.
    height_factor::T = convert(T, 1 / 2)
end

"""
    optimize_kernel_hp_separately(
        alg_trait::ExtensionPkgs,
        ref_dek::DEKernel,
        ms_trait::ModelSelection,
        model::Union{LazyGP,GPData},
        config::HyperparameterInferenceInfo,
        solver_config::SolverConfig,
        optim_config::KernelOptimConfig{T},
    ) where T <: Real

The two-stage approach to optimizing the DE kernel hyperparameters.
"""
function optimize_kernel_hp_separately(
        alg_trait::ExtensionPkgs,
        ref_dek::DEKernel,
        ms_trait::ModelSelection,
        model::Union{LazyGP, GPData},
        config::HyperparameterInferenceInfo,
        solver_config::SolverConfig,
        optim_config::KernelOptimConfig{T},
    ) where {T <: Real}

    # type conversion.
    a_lb = optim_config.a_lb
    a_ub = optim_config.a_ub
    κ_ub = optim_config.κ_ub
    width_factor = optim_config.width_factor
    height_factor = optim_config.height_factor
    N_initials_κ = optim_config.N_initials_κ

    # generate initial guesses.
    κ0s = LinRange(zero(T), κ_ub, N_initials_κ)

    # # Optimize stationary kernel.
    sk_vars, sk_star = optimize_kernel_hp(
        alg_trait,
        ref_dek.canonical,
        MarginalLikelihood(),
        model,
        config,
        solver_config,
        optim_config,
    )
    sk = createkernel(ref_dek.canonical, sk_vars[begin])

    # # adjust bandwidth.
    a_star, _, status = findbandwidth(
        sk, width_factor, a_lb, a_ub;
        h_sk = height_factor,
    )
    if !status
        a_star = sk.a
    end

    # # optimize κ.
    costfunc, Bs_info = create_hp_cost(
        resolvetrait(config),
        ms_trait,
        model,
        ref_dek,
        config,
    )

    # initial guesses.
    p0s = collect([a_star; x;] for x in κ0s)

    # clamp the dek's bandwidth to a_star
    lbs = [a_star; zero(T)]
    ubs = [a_star; κ_ub]

    # optimize.
    dek_vars, dek_star = run_optimizer(
        alg_trait,
        costfunc, lbs, ubs, solver_config, p0s,
    )

    # clean up.
    free_hp_cost(Bs_info)

    return dek_vars, dek_star, sk_vars, sk_star
end

function optimize_kernel_hp_separately_timing(
        alg_trait::ExtensionPkgs,
        ref_dek::DEKernel,
        ms_trait::ModelSelection,
        model::Union{LazyGP, GPData},
        config::HyperparameterInferenceInfo,
        solver_config::SolverConfig,
        optim_config::KernelOptimConfig{T},
    ) where {T <: Real}

    # type conversion.
    a_lb = optim_config.a_lb
    a_ub = optim_config.a_ub
    κ_ub = optim_config.κ_ub
    width_factor = optim_config.width_factor
    height_factor = optim_config.height_factor
    N_initials_κ = optim_config.N_initials_κ

    # generate initial guesses.
    κ0s = LinRange(zero(T), κ_ub, N_initials_κ)

    # # Optimize stationary kernel.
    println("Stationary kernel:")
    sk_vars, sk_star = optimize_kernel_hp(
        alg_trait,
        ref_dek.canonical,
        MarginalLikelihood(),
        model,
        config,
        solver_config,
        optim_config,
    )
    @time optimize_kernel_hp(
        alg_trait,
        ref_dek.canonical,
        MarginalLikelihood(),
        model,
        config,
        solver_config,
        optim_config,
    )
    sk = createkernel(ref_dek.canonical, sk_vars[begin])

    # # adjust bandwidth.
    a_star, _, status = findbandwidth(
        sk, width_factor, a_lb, a_ub;
        h_sk = height_factor,
    )
    if !status
        a_star = sk.a
    end

    # # optimize κ.
    costfunc, Bs_info = create_hp_cost(
        resolvetrait(config),
        ms_trait,
        model,
        ref_dek,
        config,
    )

    # initial guesses.
    p0s = collect([a_star; x;] for x in κ0s)

    # clamp the dek's bandwidth to a_star
    lbs = [a_star; zero(T)]
    ubs = [a_star; κ_ub]

    # optimize.
    println("DE kernel:")
    dek_vars, dek_star = run_optimizer(
        alg_trait,
        costfunc, lbs, ubs, solver_config, p0s,
    )
    @time run_optimizer(
        alg_trait,
        costfunc, lbs, ubs, solver_config, p0s,
    )

    # clean up.
    free_hp_cost(Bs_info)

    return dek_vars, dek_star, sk_vars, sk_star
end


"""
    optimize_kernel_hp(
        alg_trait::ExtensionPkgs,
        ref_sk::StationaryKernel,
        ms_trait::ModelSelection,
        model::Union{LazyGP,GPData},
        config::HyperparameterInferenceInfo,
        solver_config::SolverConfig,
        optim_config::KernelOptimConfig,
    )

Fits the GPR hyperparameters for multi-process, distributed computing setup.
"""
function optimize_kernel_hp(
        alg_trait::ExtensionPkgs,
        ref_sk::StationaryKernel,
        ms_trait::ModelSelection,
        model::Union{LazyGP, GPData},
        config::HyperparameterInferenceInfo,
        solver_config::SolverConfig,
        optim_config::KernelOptimConfig,
    )
    # type conversion.
    a_lb = optim_config.a_lb
    a_ub = optim_config.a_ub

    # generate initial guesses.
    a0s = LinRange(a_lb, a_ub, optim_config.N_initials_a)

    # initial guesses.
    p0s = collect([x;] for x in a0s)

    optim_info = OptimContainer([a_lb;], [a_ub;], p0s)

    return optimize_kernel_hp(
        alg_trait, ref_sk, ms_trait, model,
        config, solver_config, optim_info,
    )
end

struct UseDistributedComputing end

function resolvetrait(::LazyHCostConfig)
    return UseDistributedComputing()
end

function resolvetrait(::HCostConfig)
    return nothing
end

"""
    create_hp_cost(::UseDistributedComputing, args...)
Creates the hyperparameter cost function for a distributed computing setup.
"""
function create_hp_cost(::UseDistributedComputing, args...)
    return create_hoptim_cost_dc(args...)
end

"""
    create_hp_cost(::Nothing, args...)
Creates the hyperparameter cost function for a single-process setup.
"""
function create_hp_cost(::Nothing, args...)
    return create_hoptim_cost(args...)
end


"""
    free_hp_cost(Bs_info::DData.Dinfo)

Frees up the data specified in `Bs_info`, which was used by `create_hp_cost` and `run_optimizer`.
"""
function free_hp_cost(Bs_info::DData.Dinfo)
    return free_hoptim_cost_dc(Bs_info)
end

"""
    free_hp_cost(::Nothing, args...)
There is no workers to free for a single-process setup. This function does nothing.
"""
function free_hp_cost(::Nothing)
    return nothing
end

# find a such that k_{a}(x) == k_{a_star}(x*h) == k_{any}(0) * h_sk.
# k_{any}(0) is the maximum of the RBF k_{a}(.) for any `a`.
function findbandwidth(
        sk::Union{WendlandSplineKernel, SqExpKernel},
        h::Real,
        a_lb::T,
        a_ub::T;
        h_sk::Real = convert(T, 0.5), # half maximum.
    ) where {T <: AbstractFloat}

    @assert a_lb < a_ub
    @assert h > 0
    @assert 0 < h_sk < 1

    # find the half width at half maximum
    k_max = evalkernel(zero(T), sk)
    target_k = convert(T, h_sk * k_max)

    x_ub = one(T)
    while isfinite(x_ub) && evalkernel(x_ub, sk) > target_k
        x_ub = x_ub * 2
    end
    if !isfinite(x_ub)
        return sk.a, zero(T), false # failed to find larger bandwidth.
    end
    #@show x_ub # debug

    f = xx -> (evalkernel(xx, sk) - target_k)
    x = Roots.find_zero(f, (zero(T), x_ub))

    # look for a such that we get target_k at distance x*h.

    if isapprox(h, 1)
        return sk.a, x, true
    end

    x2 = convert(T, x * h)
    g = pp -> (evalkernel(x2, updatekernel(sk, [pp;])) - target_k)

    #@show h, h_sk, a_lb, a_ub, g(a_lb), g(a_ub) # debug
    if sign(g(a_lb)) == sign(g(a_ub))
        # there is no root in the interval (a_lb, a_ub)
        return sk.a, x, false
    end

    a_star = Roots.find_zero(g, (a_lb, a_ub))

    return a_star, x, true
end
