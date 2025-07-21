module ScatteredInterpolationExt

using LazyGPR
#const GSP = LazyGPR.GSP

import ScatteredInterpolation


struct ShepardMap{IT}
    etp::IT
end

function (ftr::ShepardMap)(
        xrs::NTuple{D, AR},
    ) where {D, AR <: AbstractRange}

    return collect(
        ScatteredInterpolation.evaluate(ftr.etp, collect(x))[begin]
            for x in Iterators.product(xrs...)
    )
end

function (ftr::ShepardMap)(x::NTuple)
    return ScatteredInterpolation.evaluate(ftr.etp, collect(x))[begin]
end


function (ftr::ShepardMap)(x::AbstractVector)
    return ScatteredInterpolation.evaluate(ftr.etp, x)[begin]
end


# function LazyGPR._createitp(
#     trait::LazyGPR.UseScatteredInterpolation,
#     Xrs::NTuple{D, AR},
#     y::AbstractArray,
#     p::Real,
#     ) where {D, AR <: AbstractRange}

#     tmp = collect(Iterators.product(Xrs...))
#     X_mat = reshape(
#         collect(Iterators.flatten(X)),
#         D, length(X),
#     )
#     return LazyGPR._createitp(trait, X_mat, y, p)
# end

function LazyGPR._createitp(
        trait::LazyGPR.UseScatteredInterpolation,
        X::AbstractVector,
        y::AbstractArray,
        p::Real,
    )

    #X_mat = LazyGPR.ranges2matrix(X)
    @assert !isempty(X)
    D = length(X[begin])

    X_mat = reshape(collect(Iterators.flatten(X)), D, length(X))
    return LazyGPR._createitp(trait, X_mat, vec(y), p)
end

# all extrapolation evalates to 0.
function LazyGPR._createitp(
        ::LazyGPR.UseScatteredInterpolation,
        X_mat::Matrix{T},
        y::Vector{T},
        p::Real,
    ) where {T <: Real}

    itp_obj = ScatteredInterpolation.interpolate(
        ScatteredInterpolation.Shepard(p),
        X_mat,
        y,
    )

    return ShepardMap(itp_obj)
end

end
