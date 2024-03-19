module RieszDSPExt

using LazyGPR
#const GSP = LazyGPR.GSP

import RieszDSP
const RZ = RieszDSP


function getrwlaplacian(
    y::Array{T,D},
    order::Integer,
    N_scales::Integer,
    ) where {T <: AbstractFloat, D}

    if N_scales <= 0
        # dyadic scaling for the largest dimension.
        N_scales = round(Int, log2( maximum(size(y))))
    end

    # Riesz-wavelet analysis.
    WRY, _, a_array = RZ.rieszwaveletanalysis(y, N_scales, order)

    # find the non-cross order terms. They are an approximation to non-cross derivatives.
    inds = findall(xx->maximum(xx)==order, a_array)
    
    # sum over band responses to get the Laplacian estimate.
    L_y = sum( sum( WRY[j][s] for j in inds) for s = 1:N_scales)
    
    #L_y_finest = sum( WRY[j][begin] for j in inds)
    #return L_y, L_y_finest, WRY, residual, a_array
    return L_y
end

function LazyGPR._create_grid_warp_samples(::LazyGPR.UseRieszDSP, y_nD::AbstractArray, N_scales)
    return getrwlaplacian(y_nD, 2, N_scales)
end

# select training inputs for hyperparameter optim



end
