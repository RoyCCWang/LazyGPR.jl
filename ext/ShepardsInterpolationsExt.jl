module ShepardsInterpolationsExt

using LazyGPR

import ShepardsInterpolations

# We're only using the distance exponent == 2 case from ShepardsInterpolations.jl.
struct Shepard2Map{T <: AbstractFloat}
    state::ShepardsInterpolations.IDWState{T}
    y::Memory{T}
    X_mat::Matrix{T}
end

function (A::Shepard2Map)(
        xrs::NTuple{D, AR},
    ) where {D, AR <: AbstractRange}

    return collect(
        ShepardsInterpolations.query_idw2!(A.state, x, A.y, A.X_mat)
            for x in Iterators.product(xrs...)
    )
end

function (A::Shepard2Map)(x::NTuple)
    return ShepardsInterpolations.query_idw2!(A.etp, x, A.y, A.X_mat)
end

function (A::Shepard2Map)(x::AbstractVector)
    return ShepardsInterpolations.query_idw2!(A.state, x, A.y, A.X_mat)
end

function LazyGPR._createitp(
        trait::LazyGPR.UseShepardsInterpolations,
        X::AbstractVector,
        y::AbstractArray,
    )

    @assert !isempty(X)
    D = length(X[begin])

    X_mat = reshape(collect(Iterators.flatten(X)), D, length(X))
    return LazyGPR._createitp(trait, X_mat, vec(y))
end

# creates a copy of the inputs.
function LazyGPR._createitp(
        ::LazyGPR.UseShepardsInterpolations,
        X_mat::AbstractMatrix{T},
        y::AbstractVector{T},
    ) where {T <: AbstractFloat}
    size(X_mat, 2) == length(y) || error("Size mismatch in the observation set.")

    return Shepard2Map(
        ShepardsInterpolations.IDWState(T, length(y)),
        Memory{T}(y),
        Matrix{T}(X_mat),
    )
end

end
