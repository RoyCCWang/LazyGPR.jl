# for dek.

function computeXc!(::StationaryKernel, args...)
    return nothing
end

# assumes Xc[1:D, :] == Xa will never change. Used in hoptim.
function computeXc!(
    dek::DEKernelFamily,
    buf::DEKNeighbourhoodBuffer,
    )
    
    return Xa2Xc!(buf.Xc, buf.ϕ_X, dek.κ)
end

# using no cached warpmap quantities.
function updateK!(
    θ::PositiveDefiniteKernel, # dispatch.
    buf::NeighbourhoodBuffer, # cached items. # mutates buf.U and buf.Xc
    )

    # apply warpmap to update buf.Xc
    computeXc!(θ, buf)

    # store pair-wise dists in K.
    K, Xc = buf.U, buf.Xc
    Distances.pairwise!(getmetric(θ), K, Xc, dims = 2)

    # store the kernel matrix in K.
    K = updatekernelmatrix!(K, K, θ)
    
    return nothing
end

# for grid data.
function evalnlli!(
    ms_trait::ModelSelection,
    buf::NeighbourhoodBuffer, # mutates, buffer.
    θ::PositiveDefiniteKernel,
    s_map,
    σ²,
    )

    updateK!(θ, buf)
    U0, RqX, y = buf.U, buf.RqX, buf.y

    # construct noise model
    U, _ = applynoisemodel!(θ, s_map, U0, σ², RqX)
    # U = U0
    # for i in axes(U, 1)
    #     U[i,i] += σ²
    # end

    return evalHOcost!(ms_trait, buf.y_buf, U, y)
end


# `y_buf` is an intermediate buffer, mutates.
function evalHOcost!(::MarginalLikelihood, y_buf::Vector{T}, U::Matrix{T}, y::Vector{T}) where T <: AbstractFloat

    C = cholesky(U) # 3 ms.

    ldiv!(y_buf, C, y)
    return dot(y, y_buf) + logdet(C) # dot(y, U\y) + logdet(U)
end

function evalHOcost!(::MarginalLikelihood, C::Cholesky, c::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    return dot(y, c) + logdet(C) # dot(y, U\y) + logdet(U)
end

# `K_inv_y` is an intermediate buffer, mutates.
# U is an intermediate buffer.
function evalHOcost!(::LOOCV, K_inv_y::Vector{T}, U::Matrix{T}, y::Vector{T})::T where T <: AbstractFloat
    
    C = cholesky(U) # 3 ms.
    
    ldiv!(K_inv_y, C, y)

    K_inv = U
    I = Diagonal(LinearAlgebra.I, length(y))
    ldiv!(K_inv, C, I) #K_inv = C\I

    return eval_negative_LOOCV(K_inv, K_inv_y, y)
end

function evalHOcost!(::LOOCV, C::Cholesky, c::Vector{T}, y::Vector{T}) where T <: AbstractFloat

    I = Diagonal(LinearAlgebra.I, length(y))
    return eval_negative_LOOCV(C\I, c, y)
end

# GPML book, Eqs. 5.10 - 5.13, without constants.
function eval_negative_LOOCV(
    K_inv::Matrix{T}, # buffer.
    K_inv_y::Vector{T},
    y::Vector{T},
    ) where T <: AbstractFloat

    cost = zero(T)
    for i in eachindex(y)

        v = 1/K_inv[i,i]
        cost += log(v) + v*K_inv_y[i]^2
    end

    return cost
end

########## select data subset.

function computevariance(
    Xrs::NTuple{D, AR},
    A::Array{T,D},
    b_x::T,
    ) where {T <: AbstractFloat, D, AR <: AbstractRange}

    max_dist = b_x
    #max_dist = b_x^2

    vs = zeros(T, size(A))
    k = 0
    for xq in Iterators.product(Xrs...)

        nbs = computeRqX(
            nothing,
            Xrs, Distances.Euclidean(), max_dist, xq,
            #Xrs, Distances.SqEuclidean(), max_dist, xq,
        )
        
        k += 1
        vs[k] = var( A[i] for i in nbs ; corrected = false )
    end

    return vs
end

function computevariance(
    S::SpatialSearchContainer,
    A::Vector{T},
    b_x::T,
    ) where T <: AbstractFloat
    
    nnt, X_data = S.nnt, S.X
    
    @assert length(A) == length(X_data)

    vs = zeros(T, length(A))
    k = 0
    for xq in X_data

        nbs = NN.inrange(nnt.tree, xq, b_x, false)

        k += 1
        vs[k] = var( A[i] for i in nbs ; corrected = false )
    end

    return vs
end

function allocatebuffers(::Type{T}, ::DEKernel, M::Integer) where T <: AbstractFloat
    return Vector{DEKNeighbourhoodBuffer{T}}(undef, M)
end

function allocatebuffers(::Type{T}, ::StationaryKernel, M::Integer) where T <: AbstractFloat
    return Vector{SKNeighbourhoodBuffer{T}}(undef, M)
end

function createtrainingset(
    Xrs::NTuple{D, AR},
    M::Integer,
    vs::Array{T,D},
    θ::PositiveDefiniteKernel,
    b_x::T,
    y::Array{T,D};
    descending = true,
    ) where {T <: AbstractFloat, D, AR <: AbstractRange}
    
    @assert 1 <= M <= length(vs)
    @assert size(y) == size(vs)

    # keep track which nodes are availble to be added to the training set..
    available = trues(size(vs))
    
    # largest to smallest.
    # don't use partialsortperm since we might need more than M sorted entries, since.
    inds = sortperm(vec(vs); rev = descending)
    
    nbs_set = Vector{Vector{CartesianIndex{D}}}(undef, M)
    node_set = Vector{Int}(undef, M)
    buffers = allocatebuffers(T, θ, M)

    CIs = CartesianIndices(size(y))

    m = 0
    for node in inds

        if available[node]
            
            m += 1
            
            buffers[m], nbs = computebuffer(
                θ, Xrs, b_x, ci2coord(CIs[node], Xrs), y,
            )
            nbs_set[m] = nbs
            node_set[m] = node

            if m == M
                return buffers, node_set, nbs_set
            end

            # book keep.
            for n in nbs
                available[n] = false
            end
        end
    end

    resize!(node_set, m)
    resize!(nbs_set, m)
    resize!(buffers, m)

    return buffers, node_set, nbs_set
end

# convinence routine for diagnostics.
function node2coord(n::Integer, sz::NTuple, Xrs::NTuple{D, AR}) where {D, AR <: AbstractRange}
    CIs = CartesianIndices(sz)
    return ci2coord(CIs[n], Xrs)
end

function createtrainingset(
    X_data::Vector{AV},
    nnt::NNTree,
    M::Integer,
    vs::Vector{T},
    θ::PositiveDefiniteKernel,
    b_x::T,
    y::Vector{T};
    descending = true,
    ) where {T <: AbstractFloat, AV <: AbstractVector}
    
    @assert 1 <= M <= length(vs)
    @assert size(y) == size(vs)

    # keep track which nodes are availble to be added to the training set..
    available = trues(length(vs))
    
    # largest to smallest.
    # don't use partialsortperm since we might need more than M sorted entries, since.
    inds = sortperm(vec(vs); rev = descending)
    
    nbs_set = Vector{Vector{Int}}(undef, M)
    node_set = Vector{Int}(undef, M)
    buffers = allocatebuffers(T, θ, M)

    m = 0
    for node in Iterators.take(inds, M)

        if available[node]
            
            m += 1
            
            buffers[m], nbs = computebuffer(
                θ, X_data, nnt, b_x, X_data[node], y,
            )
            nbs_set[m] = nbs
            node_set[m] = node

            # book keep.
            for n in nbs
                available[n] = false
            end
        end
    end

    resize!(node_set, m)
    resize!(nbs_set, m)
    resize!(buffers, m)

    return buffers, node_set, nbs_set
end

# TODO this could be more memory-efficient.
function createtrainingset_both(
    S, N_neighbourhoods, vs, θ_ref, b_x, y,
    )

    buffer1, nodes1, _= createtrainingset(
        S,
        N_neighbourhoods,
        vs, θ_ref, b_x, y,
        descending = true, # highest variance first.
    )

    buffer2, nodes2, _= createtrainingset(
        S,
        N_neighbourhoods,
        vs, θ_ref, b_x, y,
        descending = false, # highest variance first.
    )

    node_list = vcat(nodes1, nodes2)
    buffer_list = vcat(buffer1, buffer2)
    tmp = unique(
        xx->xx[begin],
        zip(node_list, buffer_list)
    )
    
    return map(xx->xx[end], tmp)
end

################### evaluate hpyerparameter optimization cost function.

function updatekernel(dek_ref::DEKernelFamily, p)

    kernel_params, κ, _ = unpackparams(p, 0, dek_ref)

    return createkernel(
        dek_ref,
        createkernel(dek_ref.canonical, kernel_params...),
        κ,
    )
end

function updatekernel(θ::StationaryKernel, p)

    kernel_params, _ = unpackparams(p, 0, θ)

    return createkernel(θ, kernel_params...)
end

# single-thread, single-process.
function evalcost_hyperoptim!(ms_trait::ModelSelection, Bs, p, θ_ref, s_map, σ²::T)::T where T <: Real
    θ = updatekernel(θ_ref, p)

    return sum(
        evalnlli!(ms_trait, B, θ, s_map, σ²)
        for B in Bs
    )
end


############################# distributed computing.
# threading causes some issues with PythonCall.jl: https://github.com/JuliaPy/PythonCall.jl/issues/219
# The work-around is to set env var PYTHON_JULIACALL_HANDLE_SIGNALS=yes. The downside is that python signal handlet (ctrl-c break loop) is no longer there.
# for some reason, FLoops.jl was slower than the the single-thread `evalcost_hyperoptim`, so we go with a distributed computing approach in this package.

abstract type HyperparameterInferenceInfo end

"""
    struct HCostConfig
This is a trait data type that indicates single-process hyperparameter inferenec should be performed. 
"""
struct HCostConfig <: HyperparameterInferenceInfo end # for conventional GP.

"""
    struct LazyHCostConfig
This is a trait and config data type that indicates multi-process hyperparameter inferenec should be performed. It also specifies that the model is the lazy-evaluation GPR, as opposed to a conventional GPR.
"""
struct LazyHCostConfig{T,D} <: HyperparameterInferenceInfo
    worker_list::Vector{Int}
    N_neighbourhoods::Int # an upperbound.
    most_and_least_variance::Bool # use both most variable and least variable samples.
    V::Array{T,D}
end

function LazyHCostConfig(N_samples::Int, V)
    return LazyHCostConfig(nworkers(), N_samples, true, V)
end

#LazyGP
function create_hoptim_cost_dc(
    ms_trait::ModelSelection,
    model::LazyGP,
    θ_ref::PositiveDefiniteKernel,
    config::LazyHCostConfig,
    )

    data = getdata(model)

    return create_hoptim_cost_dc(
        ms_trait,
        data.inputs,
        θ_ref,
        data.outputs,
        config.V,
        model.b_x,
        model.s_map,
        data.σ²,
        config.N_neighbourhoods,
        config.worker_list;
        most_and_least_variance = config.most_and_least_variance,
    )
end


# packaged version of train_grid.jl
function create_hoptim_cost_dc(
    ms_trait::ModelSelection,
    S::Union{SpatialSearchContainer, NTuple},
    θ_ref::PositiveDefiniteKernel,
    y::Array{T},
    V::Array{T},
    b_x::T,
    s_map::AdjustmentMap{T},
    σ²::T,
    N_neighbourhoods::Integer,
    worker_list_in;
    most_and_least_variance = true,
    ) where T <: AbstractFloat
    
    vs = computevariance(S, V, b_x)

    tr_buffers = Vector{DEKNeighbourhoodBuffer{T}}(undef, 0)
    if most_and_least_variance

        tr_buffers = createtrainingset_both(
            S,
            N_neighbourhoods,
            vs, θ_ref, b_x, y,
        )
    else
        tr_buffers, _= createtrainingset(
            S,
            N_neighbourhoods,
            vs, θ_ref, b_x, y,
            descending = true, # highest variance first.
        )
    end
    @assert !isempty(tr_buffers)

    worker_list = worker_list_in
    if length(worker_list) > length(tr_buffers)
        # DData.scatter_array() would generate some empty arrays on some workers.
        # need to limit the number of workers so that each gets a non-empty array via DData.scatter_array() later.
        worker_list = worker_list_in[1:length(tr_buffers)]
    end
    #@show length(tr_buffers), worker_list_in, worker_list # debug.

    θ_ref_econ = geteconkernel(θ_ref)

    # put parts of `tr_buffers` on the workers.
    Bs_info = DData.scatter_array(
        :Bs_wk, tr_buffers, worker_list,
    )

    # put all of the constant quantities on the workers. 
    map(
        fetch,
        [
            DData.save_at(
                w, :constants_wk,
                (θ_ref_econ, s_map, σ²),
            ) for w in worker_list
        ]
    )

    # define this in the Main module of the workers in `worker_list`. 
    costfunc = pp->evalcost8(ms_trait, pp, Bs_info)

    return costfunc, Bs_info
end

# based on DData.dmap()
function evalcost8(ms_trait::ModelSelection, p0, Bs_info)

    tmp = map(
        fetch,
        [
            DData.get_from(
                w,
                :(
                    LazyGPR.evalcost_hyperoptim_packed!(
                        $(ms_trait), Bs_wk, $(p0), constants_wk,
                    )
                ),
            ) for w in Bs_info.workers
        ]
    )
    return sum(tmp) 
end

function evalcost_hyperoptim_packed!(ms_trait::ModelSelection, Bs, p, data)
    return evalcost_hyperoptim!(ms_trait, Bs, p, data...)
end

#clean up.
function free_hoptim_cost_dc(Bs_info)
    
    DData.unscatter(Bs_info)
    for w in Bs_info.workers
        DData.remove_from(w, :p_wk)
        DData.remove_from(w, :constants_wk)
    end

    return nothing
end

