# function assembleXmat(x_ranges::Vector{AR}) where AR <: AbstractRange

#     return reshape(
#         collect(
#             Iterators.flatten(
#                 Iterators.product(x_ranges...)
#             )
#         ),
#         length(x_ranges),
#         prod( 
#             length(x_ranges[d]) for d in eachindex(x_ranges)
#         ),
#     )
# end

# function assembleXcmat(
#     ::StationaryKernel,
#     ::Any, # doesn't use any cached objects.
#     x_ranges::Vector{AR},
#     ) where AR <: AbstractRange

#     return assembleXmat(x_ranges)
# end

# # re-compute ϕ_X.
# function assembleXcmat(
#     dek::DEKernel,
#     ::Nothing,
#     x_ranges::Vector{AR},
#     ) where AR <: AbstractRange

#     ϕ_X_local = evaldebatch(dek, x_ranges)
    
#     X_mat = assembleXmat(x_ranges)
#     Xc_mat = [X_mat; ϕ_X_local']
#     return Xc_mat
# end

# # use cached ϕ_X.
# function assembleXcmat(
#     ::DEKernel,
#     cvars::WarpmapCache,
#     x_ranges::Vector{AR},
#     ) where AR <: AbstractRange

#     ϕ_X_local = collect(
#         cvars.ϕ_X[i...]
#         for i in Iterators.product(x_ranges...)
#     )

#     X_mat = assembleXmat(x_ranges)
#     Xc_mat = [X_mat; ϕ_X_local']
#     return Xc_mat
# end


# distance(xq, X)
function assembleRqX(x_ranges, xq::AbstractVector, dist)
    
    X_mat = assembleXmat(x_ranges)
    xq_mat = reshape(xq, (length(xq),1))

    return Distances.pairwise(dist, X_mat, xq_mat, dims = 2)
end

function computekernelmatrix(X_mat, θ::PositiveDefiniteKernel)
    dist = getmetric(θ)
    Rc = Distances.pairwise(dist, X_mat, dims = 2)

    updatekernelmatrix!(Rc, Rc, θ) # overwrite Rc with results.
    return Rc
end

function updatekernelmatrix!(
    K::Matrix{T}, # mutates, output.
    Rc::Matrix{T},
    dek::DEKernelFamily,
    ) where T

    return updatekernelmatrix!(K, Rc, dek.canonical)
end

# safe to assign K to be Rc.
function updatekernelmatrix!(
    K::Matrix{T}, # mutates, output.
    Rc::Matrix{T},
    θ::StationaryKernel,
    ) where T

    @assert size(K) == size(Rc)

    #fill!(K, convert(T, Inf))
    for j in axes(K,2)
        for i in Iterators.drop(axes(K,1), j-1)
            K[i,j] = evalkernel(Rc[i,j], θ)
        end
    end

    for j in Iterators.drop(axes(K,2), 1)
        for i in Iterators.take(axes(K,1), j-1)
            K[i,j] = K[j,i]
        end
    end

    return nothing
end

##### kernel vector for query/inference

# # mutates R, output.
# function computeKq!(
#     dek::DEKernel,
#     R::Vector, # mutates. output.
#     xq,
#     Xc_mat,
#     )

#     phi_x = evalde(dek, xq)
#     xcq_mat = reshape(
#         [xq; phi_x],
#         (length(xq)+1, 1),
#     )

#     R_mat = reshape(R, (length(R),1))
#     Distances.pairwise!(
#         getmetric(dek),
#         R_mat, # mutates, output.
#         Xc_mat,
#         xcq_mat,
#         dims = 2,
#     )
#     # @assert norm(R-R_mat) < 1e-12 # debug.

#     # R_mat and R share the same underlying data in memory. Mutating R_mat mutates R.
#     # So we can pass R here, instead of vec(R_mat)
#     return computeKq!(dek.canonical, R)
# end

################ new.


function assemblematrix(
    Xrs::NTuple{D, AR},
    ) where {D, AR <: AbstractRange}

    return reshape(
        collect(
            Iterators.flatten(
                Iterators.product(Xrs...)
            )
        ),
        D,
        prod(length(Xrs[d]) for d in eachindex(Xrs)),
    )
end

function assemblematrix(
    Xrs::NTuple{D, AR},
    nbs_ci::Vector{CartesianIndex{D}},
    ) where {D, AR <: AbstractRange}

    return reshape(
        collect(
            Iterators.flatten(
                ci2coord(ci, Xrs) for ci in nbs_ci
            )
        ),
        D, length(nbs_ci),
    )
end

function assemblematrix(
    X_data::Vector{AV},
    nbs::Vector{Int},
    ) where AV <: AbstractVector

    D = length(X_data[begin])
    return reshape(
        collect(
            Iterators.flatten(
                X_data[n] for n in nbs
            )
        ),
        D, length(nbs),
    )
end

# no multiplier.
function getxa(ci::CartesianIndex, Xrs, dek::DEKernel)
    x = ci2coord(ci, Xrs)
    z = evalwarpmap(dek, x)
    return (x..., z)
end

# for caching. Used in training.
# [x; warpmap(x)]
function assembleXa(
    Xrs::NTuple{D, AR},
    nbs_ci::Vector{CartesianIndex{D}},
    dek::DEKernel,
    ) where {D, AR <: AbstractRange}

    return reshape(
        collect(
            Iterators.flatmap(
                xx->getxa(xx, Xrs, dek),
                nbs_ci
            )
        ),
        D+1, length(nbs_ci),
    )
end

function getxa(n::Integer, X::Vector{AV}, dek::DEKernel) where AV <: AbstractVector
    x = X[n]
    z = evalwarpmap(dek, x)
    return (x..., z)
end

function assembleXa(
    X_data::Vector{AV},
    nbs::Vector{Int},
    dek::DEKernel,
    ) where AV <: AbstractVector

    D = length(X_data[begin])
    return reshape(
        collect(
            Iterators.flatmap(
                xx->getxa(xx, X_data, dek),
                nbs
            )
        ),
        D+1, length(nbs),
    )
end

# when there is no cached warpmap.

function assembleXa(X, nbs, v::WarpmapCache)
    return assemblematrix(X, nbs, v.ϕ_X)
end


function getxc(xq::NTuple{D,T}, dek::DEKernel)::Vector{T} where {T, D}
    
    out = Vector{T}(undef, D+1)
    for d in Iterators.take(eachindex(out), D)
        out[d] = xq[d]
    end

    out[end] = evalde(dek, xq)
    
    return out
end

function getxc(xq::AbstractVector{T}, dek::DEKernel)::Vector{T} where T
    phi_x = evalde(dek, xq)
    return [xq; phi_x]
end


# # for querying.
# # [x κ*warpmap(x)]
# function assembleXc(
#     Xrs::NTuple{D, AR},
#     nbs_ci::Vector{CartesianIndex{D}},
#     dek::DEKernel,
#     ) where {D, AR <: AbstractRange}

#     return reshape(
#         collect(
#             Iterators.flatmap(
#                 xx->getxc(xx, Xrs, dek),
#                 nbs_ci
#             )
#         ),
#         D+1, length(nbs_ci),
#     )
# end

function appendx(ci::CartesianIndex, Xrs::NTuple, v::Real)
    return (ci2coord(ci, Xrs)..., v)
end

# function appendx(xq::AbstractVector{T}, v::T)::Vector{T} where T
#     return [xq; v]
# end

function appendx(n::Integer, X::Vector{AV}, v::T)::Vector{T} where {T, AV <: AbstractVector{T}}
    return [X[n]; v]
end


# function Xa2Xc(Xa::Matrix{T}, κ::T) where T <: AbstractFloat
#     Xc = copy(Xa)
#     Xa2Xc!(Xa, κ)
#     return Xc
# end

function Xa2Xc!(Xc::Matrix{T}, ϕ_X::Vector{T}, κ::T) where T <: AbstractFloat
    for k in axes(Xc,2)
        Xc[end,k] = κ*ϕ_X[k]
    end
    return nothing
end

# v stores the cached ϕ_X_nbs.
function assemblematrix(
    Xrs::Union{NTuple, Vector},
    nbs_ci::Vector,
    v::Array,
    )
    
    return reshape(
        collect(
            Iterators.flatten(
                appendx(ci, Xrs, v[ci])
                #(ci2coord(ci, Xrs)..., v[ci])
                for ci in nbs_ci
            )
        ),
        getdim(Xrs)+1, length(nbs_ci),
    )
end

function getdim(X::Vector{AVT}) where AVT <: Union{AbstractVector, Tuple}
    return length(X[begin])
end

function getdim(::NTuple{D, AR}) where {D, AR <: AbstractRange}
    return D
end