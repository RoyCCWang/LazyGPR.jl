function optimize_hp(
        case_label, alg_trait, ref_dek, ms_trait, model,
        config, solver_config, a_lb, a_ub, κ_ub;
        width_factor = 1,
    )

    if case_label == "1" || case_label == "2"
        dek_vars, dek_star, sk_vars, sk_star = LGP.optimize_kernel_hp_separately(
            alg_trait,
            ref_dek,
            ms_trait,
            model,
            config,
            solver_config,
            LGP.KernelOptimConfig{T}( #optim_config
                a_lb = a_lb,
                a_ub = a_ub,
                κ_ub = κ_ub,
                width_factor = convert(T, width_factor),
            ),
        )
        return dek_vars, dek_star, sk_vars, sk_star
    end

    ref_sk = ref_dek.canonical
    a0s = LinRange(a_lb, a_ub, 100)

    #stationary.
    lbs = [a_lb;]
    ubs = [a_ub;]
    p0s = collect([x;] for x in a0s)

    sk_vars, sk_star = LGP.optimize_kernel_hp(
        alg_trait,
        ref_sk,
        ms_trait,
        model,
        config,
        solver_config,
        LGP.OptimContainer(lbs, ubs, p0s),
    )

    #DEK
    lbs = [a_lb; zero(T)]
    ubs = [a_ub; κ_ub]

    a0s = collect(LinRange(a_lb, a_ub, 10))
    push!(a0s, sk_vars[begin])
    κ0s = LinRange(0, κ_ub, 10)
    p0s = collect.(
        vec(collect(Iterators.product(a0s, κ0s)))
    )

    dek_vars, dek_star = LGP.optimize_kernel_hp(
        alg_trait,
        ref_dek,
        ms_trait,
        model,
        config,
        solver_config,
        LGP.OptimContainer(lbs, ubs, p0s),
    )

    return sk_vars, sk_star, dek_vars, dek_star
end


function compute_kodak_warpmap(
        W::Array{T}, sigma_r_factor, Xrs,
    ) where {T <: AbstractFloat}

    σr = maximum(abs.(W)) * sigma_r_factor
    @show sigma_r_factor, σr

    if σr > 0 && σs > 0
        W = LocalFilters.bilateralfilter(
            W, σr, σs, 2 * round(Int, 3 * σs) + 1,
        )
    end
    warpmap = LGP.create_warp_map(
        LGP.UseInterpolations(Interpolations),
        Xrs, W,
    )
    return warpmap
end

function compute_kodak_hp(
        model,
        W::Array{T}, sigma_r_factor, Xrs, ref_can_kernel,
        model_selection_trait,
        alg_trait,
        lazy_hopt_config,
        solver_config, optim_config,
    ) where {T}

    Random.seed!(25) #Since the algorithms from Metaheuristics.jl are random, we need to reset the seed for repeatable results.

    warpmap = compute_kodak_warpmap(W, sigma_r_factor, Xrs)

    dek_ref = LGP.DEKernel(
        ref_can_kernel, warpmap, zero(T),
    )

    dek_vars, dek_star, sk_vars, sk_star = LGP.optimize_kernel_hp_separately(
        alg_trait,
        dek_ref,
        model_selection_trait,
        model,
        lazy_hopt_config,
        solver_config,
        optim_config,
    )

    return dek_vars, dek_star, sk_vars, sk_star
end

function upconvert_kodak_sk(
        sk_vars,
        worker_list, model, options,
        Xqrs,
    )

    kernel_param_sk = sk_vars[begin]
    sk = LGP.WendlandSplineKernel(
        LGP.Order2(), kernel_param_sk, 3,
    )
    @show sk

    LGP.setup_query_dc(
        worker_list, model, sk, options, nothing,
    )

    #Query: tationary kernel
    println("Timing: query stationary")
    @time out = LGP.query_dc(Xqrs, worker_list)
    mqs_sta = map(xx -> xx[begin], out)
    vqs_sta = map(xx -> xx[begin + 1], out)

    LGP.free_query_dc(worker_list)

    return mqs_sta, vqs_sta
end


function upconvert_kodak_dek(
        W, sigma_r_factor, Xrs,
        dek_vars,
        worker_list, model, options,
        Xqrs,
    )

    warpmap = compute_kodak_warpmap(W, sigma_r_factor, Xrs)

    #create kernel.
    kernel_param, κ = dek_vars[begin], dek_vars[end]

    canonical_kernel = LGP.WendlandSplineKernel(
        LGP.Order2(), kernel_param, 3,
    )
    dek = LGP.DEKernel(canonical_kernel, warpmap, κ)
    @show dek.canonical, dek.κ

    #query.
    cvars = LGP.computecachevars(dek, model)
    LGP.setup_query_dc(
        worker_list, model, dek, options, cvars,
    )

    #Query: tationary kernel
    println("Timing: query dek")
    @time out = LGP.query_dc(Xqrs, worker_list)
    mqs_dek = map(xx -> xx[begin], out)
    vqs_dek = map(xx -> xx[begin + 1], out)

    LGP.free_query_dc(worker_list)

    return mqs_dek, vqs_dek
end
