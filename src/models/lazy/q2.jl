# assumes Xc_mat is already updated. Therefore, call this after updateK!().
# need to first compute k(xc_query, X) instead of k(x_query, X).
function computeKq!(
    dek::DEKernel,
    R::Vector, # mutates. overwrites. output.
    xq,
    Xc_mat,
    valids::Union{BitVector,Nothing},
    )

    xcq_mat = reshape(
        getxc(xq, dek),
        (length(xq)+1, 1),
    )

    R_mat = reshape(R, (length(R),1))
    Distances.pairwise!(
        getmetric(dek),
        R_mat, # mutates, output.
        Xc_mat,
        xcq_mat,
        dims = 2,
    )
    # @assert norm(R-R_mat) < 1e-12 # debug.

    # R_mat and R share the same underlying data in memory. Mutating R_mat mutates R.
    # So we can pass R here, instead of vec(R_mat)
    R = computeKq!(dek.canonical, R)
    return truncatearray(valids, R)
end

function truncatearray(::Nothing, x::Array)
    return x
end

function truncatearray(valids::BitArray, x::Array)
    return x[valids]
end

# mutates R, output and input. R is expected to be R_xq_X.
function computeKq!(
    θ::StationaryKernel,
    R::Vector, # mutates. input and output.
    args...)

    for i in eachindex(R)
        R[i] = evalkernel(R[i], θ)
    end

    return R
end

# when we don't use any warpmap cache.
function computequerybuffer(::Nothing, args...)
    return computebuffer(args...)
end

# use warpmap cache, data sampled from a grid.
function computequerybuffer(
    v::WarpmapCache,
    dek::DEKernel,
    Xrs::NTuple{D, AR},
    b_x::T,
    xq::Union{Tuple, AbstractVector},
    y::Array,
    ) where {T <: AbstractFloat, D, AR <: AbstractRange}
    
    RqX, nbs_ci = computeRqX(
        Xrs, getmetric(dek), b_x, xq,
    )
    #serialize("debug", (Xrs, nbs_ci, v.ϕ_X))
    Xa = assemblematrix(Xrs, nbs_ci, v.ϕ_X)
    ϕ_X = v.ϕ_X[nbs_ci]

    N = length(nbs_ci)
    U = zeros(T, N, N)

    return DEKNeighbourhoodBuffer(
        U, Xa, vec(ϕ_X), RqX, y[nbs_ci], zeros(T, N),
    ), nbs_ci
end

# use warpmap cache, generic data.
function computequerybuffer(
    v::WarpmapCache,
    dek::DEKernel,
    X_data::Vector{AV},
    nnt::NNTree,
    b_x::T,
    xq::Union{Tuple, AbstractVector},
    y::Array,
    ) where {T <: AbstractFloat, AV <: AbstractVector}
    
    RqX, nbs, _ = computeRqX(nnt, X_data, getmetric(dek), b_x, xq)
    Xa = assemblematrix(Xrs, nbs_ci, v.ϕ_X)
    ϕ_X = v.ϕ_X[nbs]

    N = length(nbs)
    U = zeros(T, N, N)

    return DEKNeighbourhoodBuffer(
        U, Xa, vec(ϕ_X), RqX, y[nbs], zeros(T, N),
    ), nbs
end


# ############## query..


function lazyquery2(
    xq::Union{NTuple, AbstractVector}, # needt o be able to reshape(). Tuples won't work. StaticArrays can.
    model::LazyGP{T},
    θ::PositiveDefiniteKernel,
    op::QueryOptions,
    cached_ϕ::Union{Nothing, WarpmapCache},
    )::Tuple{T,T} where T <: AbstractFloat
    
    b_x, s_map = model.b_x, model.s_map
    data = getdata(model)

    # get neighbourhood quantities and allocate storage.
    buf, _ = computequerybuffer(
        cached_ϕ,
        θ, data.inputs, b_x, xq, data.outputs,
    )
    
    # update matrix.
    updateK!(θ, buf)
    U0, RqX, y, y_buf = buf.U, buf.RqX, buf.y, buf.y_buf

    # construct noise model
    U, valid_diag_entries = applynoisemodel!(θ, s_map, U0, data.σ², RqX)

    # # Predictive posterior
    # k(xq, X), updated in case θ is a dek (dimensional expansion kernel).
    kq = computeKq!(θ, RqX, xq, buf.Xc, valid_diag_entries)
    
    #@show xq, U
    
    # # predictive posterior
    
    # ## option 1: direct linear solve.
    # pred_mean = dot(kq, U\y)
    # pred_var = dot(kq, U\kq)

    # ## option 2: cholesky first, then inverse.
    # C = cholesky(U)
    # pred_mean = dot(kq, C\y)
    # pred_var = dot(kq, C\y)

    ## option 3: cholesky first, ldiv!, then inverse.
    C = cholesky(U)
    # ldiv!(C, y) # overwrites y.
    # pred_mean = dot(kq, y)

    # ldiv!(y, C, kq) # overwrites y.
    # pred_var = dot(kq, y)

    pred_mean = computemean(op.compute_mean, y_buf, y, kq, C)
    pred_var = evalkernel(zero(T), θ) - computevar_term2!(op.compute_variance, y_buf, kq, C)

    # pred_mean = computemean(op.compute_mean, kq, y, y, C)
    # pred_var = evalkernel(zero(T), θ) - computevar_term2!(op.compute_variance, y, kq, C)

    return pred_mean, pred_var
end

############ single-core, single-thread, frontend

# the output is a multi-dim array that preserves the shape of the grid in `Xqrs`.
function batchqueryranges(Xqrs::NTuple{D, AR}, args...) where {D, AR <: AbstractRange}
    return batchquery(Iterators.product(Xqrs...), args...)
end

# Xq should be an iterator for NTuple or AbstractVector.
function batchquery(
    Xq, #::Vector{AVT},
    model::LazyGP,
    θ::PositiveDefiniteKernel,
    options::QueryOptions,
    cvars::CacheInfo,
    ) #where {AVT <: Union{AbstractVector, Tuple}}

    return collect(
        lazyquery2( xq, model, θ, options, cvars )
        for xq in Xq
    )
end


############## distributed computing, frontend

# # too dangerous for beginner or careless users
# # convience. The calling scope must not use a previously returned function 
# function setup_queryfunc_dc(worker_list, args...)
    
#     setup_query_dc(worker_list, args...)

#     return xx->query_dc(xx, worker_list)
# end

"""
    setup_query_dc(
        worker_list,
        model::LazyGP,
        θ::PositiveDefiniteKernel,
        options::QueryOptions,
        cvars::CacheInfo,
    )

Loads the neccessary data to the workers in `worker_list` in preparation to run batch query for the lazy-evaluation GPR model, `model`, under a distributed computing setup.
"""
function setup_query_dc(
    worker_list,
    model::LazyGP,
    θ::PositiveDefiniteKernel,
    options::QueryOptions,
    cvars::CacheInfo,
    )

    # put all of the constant quantities on the workers. 
    map(
        fetch,
        [
            DData.save_at(
                w, :lazyquery_constants_wk,
                (model, θ, options, cvars),
            ) for w in worker_list
        ]
    )
    
    #queryfunc = xx->query_dc(xx, worker_list)
    #return queryfunc
    return nothing
end

function batchquery_packed(Xqs, data)
    return batchquery(Xqs, data...)
end

"""
    query_dc(Xq::Vector, worker_list::Vector{Int})

Query the lazy-evaluation GPR model that was setup by a previous call to `setup_query_dc`.

The user should call `free_query_dc` after this function.
"""
function query_dc(Xq::Vector, worker_list::Vector{Int})

    # put parts of the query inputs on the workers.
    DData.scatter_array(
        :Xq_wk, Xq, worker_list,
    )

    tmp = map(
        fetch,
        [
            DData.get_from(
                w,
                :(
                    LazyGPR.batchquery_packed(
                        Xq_wk, lazyquery_constants_wk,
                    )
                ),
            ) for w in worker_list
        ]
    )
    out = collect( Iterators.flatten(tmp) )
    return out
end

"""
    query_dc(Xqrs::NTuple{D, AR}, worker_list::Vector{Int}) where {D, AR <: AbstractRange}
"""
function query_dc(Xqrs::NTuple{D, AR}, worker_list::Vector{Int}) where {D, AR <: AbstractRange}
    
    # Xq = vec(collect( Iterators.product(Xqrs...) ))
    it_prod = Iterators.product(Xqrs...)

    N_workers = nworkers()
    Nq_lb_per_worker = ceil(Int, length(it_prod)/N_workers)
    it_par = Iterators.partition(
        it_prod, Nq_lb_per_worker,
    )

    # an array of iterators. Haven't collected yet.
    it_array = collect(
        Iterators.take(Iterators.drop(it_par, k-1),1)
        for k = 1:length(it_par)
    )

    tmp = map(
        fetch,
        [
            DData.get_from(
                w,
                quote 
                    Xq_wk = first($(it_p_k)) # can't get around allocating due tot he nature of Iterators.partition(Iterators.product())
                    LazyGPR.batchquery_packed(
                        Xq_wk, lazyquery_constants_wk,
                    )
                end,
            ) for (w, it_p_k) in zip(worker_list, it_array)
        ]
    )
    out = collect( Iterators.flatten(tmp) )
    return out
end

"""
    free_query_dc(worker_list)

Free the data used by the `setup_query_dc` and `query_dc()` functions.
"""
function free_query_dc(worker_list)

    for w in worker_list
        DData.remove_from(w, :Xq_wk)
        DData.remove_from(w, :lazyquery_constants_wk)
    end

    return nothing
end