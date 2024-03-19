# assemble the neighbourhood non-hyperparameter quantities:
# see NeighbourhoodBuffer in types.jl
# kernel matrix/ pair-wise distance matrix buffer

function getcomparedist(::Distances.Euclidean, b_x)
    return b_x
end

function getcomparedist(::Distances.SqEuclidean, b_x)
    return b_x^2
end

########## for data sampled from a grid.

function getCIinterval(x_d::T, rd::AbstractRange, b_x::T) where T <: Real
    Δt = step(rd)
    t0 = rd[begin]

    r_lb = x_d - b_x
    r_ub = x_d + b_x
    
    # +1 since the formula is for 0-indexing n: t = t0 + n*Δt.
    n_lb = max(floor(Int, (r_lb - t0)/Δt) +1, 1)
    n_ub = min(ceil(Int, (r_ub - t0)/Δt) +1, length(rd))

    return n_lb:n_ub
end

function findsubgrid3(xrs::NTuple{D, AR}, b_x::Real, xq) where {D, AR <: AbstractRange}

    return ntuple(
        dd->getCIinterval(xq[begin+dd-1], xrs[dd], b_x),
        D,
    )
end

function computeRqX(
    Xrs::NTuple{D, AR},
    args...
    ) where {D, AR <: AbstractRange}
    
    return computeRqX(StoreDist(), Xrs, args...)
end

function computeRqX(
    storage_trait::Union{Nothing, StoreDist},
    Xrs::NTuple{D, AR},
    metric,
    b_x::Real,
    xq::Union{Tuple, AbstractVector},
    ) where {D, AR <: AbstractRange}
    
    # find box bound around the neighbourhood.
    box_ranges = findsubgrid3(Xrs, b_x, xq)
    box_CIs = CartesianIndices(box_ranges)
    
    # filter according to neighbourhood radius b_x.
    max_dist = getcomparedist(metric, b_x)
    
    return computeRqXinner(
        storage_trait,
        box_CIs, Xrs, metric, xq, max_dist,
    )
end

function ci2coord(ci::CartesianIndex{D}, Xrs::NTuple{D, AR}) where {D, AR <: AbstractRange}
    return ntuple(nn->Xrs[nn][ci[nn]], length(ci))
end

function evaldist(
    ::Distances.Euclidean,
    ci::CartesianIndex{D},
    Xrs::NTuple{D, AR},
    xq) where {D, AR <: AbstractRange}

    return sqrt(evaldist(Distances.SqEuclidean(), ci, Xrs, xq))
end

function evaldist(
    ::Distances.SqEuclidean,
    ci::CartesianIndex{D},
    Xrs::NTuple{D, AR},
    xq) where {D, AR <: AbstractRange}

    return sum(
        (Xrs[d][ci[d]] - x)^2
        for (d, x) in Iterators.enumerate(xq)
    )
end

function computeRqXinner(
    ::StoreDist,
    box_CIs::CartesianIndices{D},
    Xrs,
    metric,
    xq,
    max_dist::T,
    ) where {T <: Real, D}

    RqX = Vector{T}(undef, length(box_CIs))
    nbs = Vector{CartesianIndex{D}}(undef, length(box_CIs))

    k = 0
    #for (i, ci) in Iterators.enumerate(box_CIs)
    for ci in box_CIs
        #r = Distances.evaluate(metric, Tuple(ci), xq)
        #r = Distances.evaluate(metric, ci2coord(ci, Xrs), xq) # 25% slower than above.
        r = evaldist(metric, ci, Xrs, xq) # avoids intermediate tuple allocation.
        if r < max_dist
            #
            k += 1
            RqX[k] = r
            #nbs[k] = i
            nbs[k] = ci
        end
    end
    resize!(RqX, k)
    resize!(nbs, k)
    
    return RqX, nbs
    #return RqX, box_CIs[nbs]
end


function computeRqXinner(
    ::Nothing,
    box_CIs::CartesianIndices{D},
    Xrs,
    metric,
    xq,
    max_dist::T,
    ) where {T <: Real, D}

    nbs = Vector{CartesianIndex{D}}(undef, length(box_CIs))

    k = 0
    for ci in box_CIs
        r = Distances.evaluate(metric, ci2coord(ci, Xrs), xq) # 25% slower than above.
        if r < max_dist
            #
            k += 1
            nbs[k] = ci
        end
    end
    resize!(nbs, k)
    
    # # slower!
    # nbs_nD = collect(
    #     Iterators.filter(
    #         ci->(Distances.evaluate(metric, ci2coord(ci, Xrs), xq) < max_dist ),
    #         box_CIs
    #     )
    # )
    # nbs = vec(nbs_nD)

    return nbs
end

# warpmaps not evaluated in this function.
# compute cache (non-hyperparams) at xq. used for training.
function computebuffer(
    dek::DEKernel,
    Xrs::NTuple{D, AR},
    b_x::T,
    xq::Union{Tuple, AbstractVector},
    y::Array,
    ) where {T <: AbstractFloat, D, AR <: AbstractRange}
    
    RqX, nbs_ci = computeRqX(
        Xrs, getmetric(dek), b_x, xq,
    )
    Xa = assembleXa(Xrs, nbs_ci, dek)
    ϕ_X = vec(Xa[end,:])

    N = length(nbs_ci)
    U = zeros(T, N, N)

    return DEKNeighbourhoodBuffer(
        U, Xa, ϕ_X, RqX, y[nbs_ci], zeros(T, N),
    ), nbs_ci
end

# compute cache (non-hyperparams) at xq. used for training.
function computebuffer(
    θ::StationaryKernel,
    Xrs::NTuple{D, AR},
    b_x::T,
    xq::Union{Tuple, AbstractVector},
    y::Array,
    ) where {T <: AbstractFloat, D, AR <: AbstractRange}
    
    RqX, nbs_ci = computeRqX(
        Xrs, getmetric(θ), b_x, xq,
    )
    X_mat = assemblematrix(Xrs, nbs_ci)

    N = length(nbs_ci)
    U = zeros(T, N, N)
    
    return SKNeighbourhoodBuffer(
        U, X_mat, RqX, y[nbs_ci], zeros(T, N),
    ), nbs_ci
end


####################### for data that isn't sampled from a grid.


function setupNNtree(X::Vector{AV}) where AV <: AbstractVector
    @assert !isempty(X)

    X_mat = reshape(
        collect(Iterators.flatten(X)),
        length(X[begin]), length(X),
    )

    return NNTree(NN.KDTree(X_mat))
end

# faster than reshape(collect(a), D, 1). in Julia v.1.10.
function createcolmat(a::NTuple{D,T})::Matrix{T} where {T, D}
    out = Matrix{T}(undef, D,1)
    for d = 1:D
        out[d] = a[d]
    end
    return out
end

function createcolmat(a::AbstractVector)
    return reshape(a, length(a), 1)
end

function computeRqX(
    nnt::NNTree,
    X_data::Vector{AV},
    metric,
    b_x::Real,
    xq::Union{Tuple, AbstractVector},
    ) where AV <: AbstractVector
    
    # find box bound around the neighbourhood.
    nbs = NN.inrange(nnt.tree, xq, b_x, false)
    
    X_mat = assemblematrix(X_data, nbs)
    RqX = Distances.pairwise(
        metric,
        X_mat,
        createcolmat(xq),
        dims = 2,
    )

    return vec(RqX), nbs, X_mat
end

# warpmaps not evaluated in this function.
# compute cache (non-hyperparams) at xq. used for training.
function computebuffer(
    dek::DEKernel,
    X_data::Vector{AV},
    nnt::NNTree,
    b_x::T,
    xq::Union{Tuple, AbstractVector},
    y::Array,
    ) where {T <: AbstractFloat, AV <: AbstractVector}
    
    RqX, nbs, _ = computeRqX(nnt, X_data, getmetric(dek), b_x, xq)
    Xa = assembleXa(X_data, nbs, dek)
    ϕ_X = vec(Xa[end,:])

    N = length(nbs)
    U = zeros(T, N, N)

    return DEKNeighbourhoodBuffer(
        U, Xa, ϕ_X, RqX, y[nbs],  zeros(T, N),
    ), nbs
end

# compute cache (non-hyperparams) at xq. used for training.
function computebuffer(
    θ::StationaryKernel,
    X_data::Vector{AV},
    nnt::NNTree,
    b_x::T,
    xq::Union{Tuple, AbstractVector},
    y::Array,
    ) where {T <: AbstractFloat, AV <: AbstractVector}
    
    RqX, nbs, X_mat = computeRqX(nnt, X_data, getmetric(θ), b_x, xq)
    
    N = length(nbs)
    U = zeros(T, N, N)
    
    return SKNeighbourhoodBuffer(
        U, X_mat, RqX, y[nbs], zeros(T, N),
    ), nbs
end
