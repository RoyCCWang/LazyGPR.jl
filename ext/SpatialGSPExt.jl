module SpatialGSPExt

using LazyGPR
#const GSP = LazyGPR.GSP

import SpatialGSP
const GSP = SpatialGSP


# all extrapolation evalates to 0.
function LazyGPR._create_grid_warp_samples(
    ::LazyGPR.UseSpatialGSP,
    x_nD::AbstractArray{T},
    warp_config::GSP.WarpConfig,
    smooth_iters::Integer,
    ) where T
    
    warp_samples, G = GSP.get_grid_warp_samples(x_nD, warp_config)
    
    # additional smoothing.
    if smooth_iters > 0
        A = GSP.create_adjacency(G)
        for _ = 1:smooth_iters
            
            warp_samples = A*warp_samples
        end
    end

    # setup interpolator/extrapolator.
    return reshape(warp_samples, size(x_nD))
end

function LazyGPR._create_warp_samples(
    ::LazyGPR.UseSpatialGSP,
    ::LazyGPR.AxisGraph,
    X::Vector{AV},
    y::Vector{T},
    axis_config::GSP.AxisSearchConfig,
    warp_config::GSP.WarpConfig,
    smooth_iters::Integer,
    ) where {T <: Real, AV <: AbstractVector{T}}
    
    # warp samples.
    G = GSP.create_axis_graph(axis_config, X)
    TL = GSP.create_rwlaplacian(G)
    W = GSP.get_warp_samples(y, TL, warp_config)

    # additional smoothing on the warp samples.
    if smooth_iters > 0
        A = GSP.create_adjacency(G)
        for _ = 1:smooth_iters
            
            W = A*W
        end
    end

    return W
end

function LazyGPR._create_warp_samples(
    ::LazyGPR.UseSpatialGSP,
    ::LazyGPR.KNNGraph,
    X::Vector{AV},
    y::Vector{T},
    knn_config::GSP.KNNConfig,
    warp_config::GSP.WarpConfig,
    smooth_iters::Integer,
    ) where {T <: Real, AV <: AbstractVector{T}}
    
    # warp samples.
    W, G = GSP.get_knn_warp_samples(X, y, warp_config, knn_config)

    # additional smoothing on the warp samples.
    if smooth_iters > 0
        A = GSP.create_adjacency(G)
        for _ = 1:smooth_iters
            
            W = A*W
        end
    end

    return W
end

end