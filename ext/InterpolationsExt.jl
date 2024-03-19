module InterpolationsExt

using LazyGPR
#const GSP = LazyGPR.GSP

import Interpolations


struct InterpolationsMap{IT}
    etp::IT
end

function (ftr::InterpolationsMap)(
    xrs::NTuple{D, AR},
    ) where {D, AR <: AbstractRange}
    
    return collect(
        ftr.etp(i...)
        #ftr.etp(i)
        for i in Iterators.product(xrs...)
    )
end

# if x::Vector, then this is slow.
function (ftr::InterpolationsMap)(x::Union{NTuple{D,T}, AbstractVector{T}})::T where {D, T <: AbstractFloat}
    return ftr.etp(x...)
    #return ftr.etp(x)
end


# all extrapolation evalates to 0.
function LazyGPR._createitp(
    ::LazyGPR.UseInterpolations,
    Xrs::NTuple{D, AR},
    A::AbstractArray{T},
    ) where {T, D, AR <: AbstractRange}
    
    itp = Interpolations.interpolate(
        A,
        Interpolations.BSpline( 
            Interpolations.Cubic(    
                Interpolations.Line(Interpolations.OnGrid()),
            ),
        ),
    )
    #itp = Interpolations.interpolate(real.(A), Interpolations.BSpline(Interpolations.Quadratic(Interpolations.Line(Interpolations.OnGrid()))))
    scaled_itp = Interpolations.scale(
        itp, Xrs...,
    )
    scaled_etp = Interpolations.extrapolate(
        scaled_itp, zero(T),
    ) # zero outside interp range.

    return InterpolationsMap(scaled_etp)
end



end