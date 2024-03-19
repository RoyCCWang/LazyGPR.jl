
#### grid data.


"""
    struct UseSpatialGSP
This is a trait data type that indicates SpatialGSP.jl should be used for constructing the warp samples.
Usage:
```
import SpatialGSP as GSP
warp_graph_trait = UseSpatialGSP(GSP)
```
"""
struct UseSpatialGSP <: ExtensionPkgs
    pkg::Module
end

"""
    create_grid_warp_samples(
        alg_trait::UseSpatialGSP,
        y_nD::AbstractArray{T},
        warp_config;
        smooth_iters::Integer = 0,
    ) where T

Returns an array of warp samples, constructed using the Bernstein filtering method with a unit grid graph.

In the current implementation, `warp_config` should be a data type of `SpatialGSP.WarpConfig`.

As a post-processing step, `smooth_iters` performs smoothing by applying the adjacency matrix to the warp samples `smooth_iters` times.
"""
function create_grid_warp_samples(
    alg_trait::UseSpatialGSP,
    y_nD::AbstractArray{T},
    warp_config;
    smooth_iters::Integer = 0,
    ) where T

    alg = alg_trait.pkg
    ext = Base.get_extension(@__MODULE__, :SpatialGSPExt)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext) && alg == ext.SpatialGSP
    if !proceed_flag
        loaderr_ext_warp_samples_pkg()
    end
    
    # see SpatilGSPExt.jl
    return _create_grid_warp_samples(alg_trait, y_nD, warp_config, smooth_iters)
end

"""
    struct UseRieszDSP
This is a trait data type that indicates RieszDSP.jl should be used for constructing the warp samples.
Usage:
```
import RieszDSP as RZ
warp_graph_trait = UseRieszDSP(RZ)
```
"""
struct UseRieszDSP <: ExtensionPkgs
    pkg::Module
end

"""
    create_grid_warp_samples(
        alg_trait::UseRieszDSP,
        y_nD::AbstractArray{T};
        N_scales = 0
    ) where T

Returns an array of warp samples, constructed using the higher-order Riesz-wavelet transform, implemented in the RieszDSP.jl package.

`N_scales` is the number of wavelet bands to use. The default value of `0` means the following is used:
```
N_scales = round(Int, log2( maximum(size(y_nD))))
```
"""
function create_grid_warp_samples(
    alg_trait::UseRieszDSP,
    y_nD::AbstractArray{T};
    N_scales = 0
    ) where T

    alg = alg_trait.pkg
    ext = Base.get_extension(@__MODULE__, :RieszDSPExt)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext) && alg == ext.RieszDSP
    if !proceed_flag
        loaderr_ext_warp_samples_pkg()
    end
    
    return _create_grid_warp_samples(alg_trait, y_nD, N_scales)
end

##### warpmap construction from scattered data.

abstract type GraphTrait end

"""
    struct AxisGraph
This is a trait data type that indicates the axis-search graph construction method should be used.
"""
struct AxisGraph <: GraphTrait end

"""
    struct KNNGraph
        This is a trait data type that indicates the k-nearest neighbors graph construction method, with forced symmetry to generate undirected graphs, should be used.
"""
struct KNNGraph <: GraphTrait end

"""
    function create_warp_samples(
        alg_trait::UseSpatialGSP,
        graph_trait::GraphTrait,
        X::Vector{AV},
        y::Vector{T},
        graph_config,
        warp_config;
        smooth_iters::Integer = 0,
    ) where {T <: Real, AV <: AbstractVector{T}}

Returns an array of warp samples, constructed using the Bernstein filtering method with a graph construction method specified by `graph_trait`.

In the current implementation, `warp_config` should be a data type of `SpatialGSP.WarpConfig`, and `graph_trait` can be of the dispatch types `AxisGraph` and `KNNGraph`.

As a post-processing step, `smooth_iters` performs smoothing by applying the adjacency matrix to the warp samples `smooth_iters` times.
"""
function create_warp_samples(
    alg_trait::UseSpatialGSP,
    graph_trait::GraphTrait,
    X::Vector{AV},
    y::Vector{T},
    graph_config,
    warp_config;
    smooth_iters::Integer = 0,
    ) where {T <: Real, AV <: AbstractVector{T}}

    alg = alg_trait.pkg
    ext = Base.get_extension(@__MODULE__, :SpatialGSPExt)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext) && alg == ext.SpatialGSP
    if !proceed_flag
        loaderr_ext_warp_samples_pkg()
    end
    
    return _create_warp_samples(alg_trait, graph_trait, X, y, graph_config, warp_config, smooth_iters)
end

#### for use with the interpolations.jl extension.

"""
    struct UseInterpolations
This is a trait data type that indicates Interpolations.jl should be used to construct a warp function from a set of warp samples.
Usage:
```
import Interpolations
itp_trait = UseInterpolations(Interpolations)
```
"""
struct UseInterpolations <: ExtensionPkgs
    pkg::Module
end

# modified from minute 18 from https://www.youtube.com/watch?v=TiIZlQhFzyk
function create_warp_map(
    #alg::Module,
    alg_trait::UseInterpolations,
    Xrs::NTuple{D, AR},
    A::AbstractArray,
    args...,
    ) where {D, AR <: AbstractRange}

    alg = alg_trait.pkg
    ext_interpolations = Base.get_extension(@__MODULE__, :InterpolationsExt)
    # if !isnothing(ext_interpolations) && alg == ext_interpolations.Interpolations
    #     return _createitp(alg_trait, Xrs, A, args...)
    # end
    #return _createitp(nothing, Xrs, A)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext_interpolations) && alg == ext_interpolations.Interpolations
    if !proceed_flag
        loaderr_ext_itp_pkg()
    end
    
    return _createitp(alg_trait, Xrs, A, args...)
end

#### shared methods.

function _createitp(::Nothing, args...)
    return loaderr_ext_itp_pkg()
end

function loaderr_ext_itp_pkg()
    error("Please load a supported regression or interpolation package in your active Julia environment in order to use createitp(). The supported packages are: Interpolations.jl for interpolation on a grid; ScatteredInterpolations.jl for scattereed interpolations in arbitrary dimensions.")
end

function _create_grid_warp_samples(::Nothing, args...)
    return loaderr_ext_warp_sample_pkg()
end

function _create_warp_samples(::Nothing, args...)
    return loaderr_ext_warp_sample_pkg()
end

function loaderr_ext_warp_samples_pkg()
    error("Please load a supported warp sample generation package in your active Julia environment. Use SpatialGSP.jl for both scattered data and data on a grid, and RieszDSP.jl for data on a grid.")
end

#### for use with the ScatteredInterpolation.jl extension.

"""
    struct UseScatteredInterpolation
This is a trait data type that indicates ScatteredInterpolation.jl should be used to construct a warp function from a set of warp samples.
Usage:
```
import ScatteredInterpolation
itp_trait = UseScatteredInterpolation(ScatteredInterpolation)
```
"""
struct UseScatteredInterpolation <: ExtensionPkgs
    pkg::Module
end

# modified from minute 18 from https://www.youtube.com/watch?v=TiIZlQhFzyk
function create_warp_map(
    alg_trait::UseScatteredInterpolation,
    X::AbstractArray,
    y::AbstractVector,
    shepard_p::Real,
    )

    alg = alg_trait.pkg
    ext_si = Base.get_extension(@__MODULE__, :ScatteredInterpolationExt)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext_si) && alg == ext_si.ScatteredInterpolation
    if !proceed_flag
        loaderr_ext_itp_pkg()
    end
    
    return _createitp(alg_trait, X, y, shepard_p)
end

