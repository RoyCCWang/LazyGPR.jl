# traits

struct Enable end
struct Disable end

const BinarySelection = Union{Enable, Disable}

# for lazy querying.
function computemean!(
        ::Enable,
        buf::Vector{T}, # buffer.
        y::Vector{T},
        kq::Vector{T},
        C,
    )::T where {T}

    ldiv!(buf, C, y)
    return dot(kq, buf)
end

function computemean!(::Disable, ::Vector{T}, args...)::T where {T}
    return convert(T, NaN)
end

# mutates buf, buffer.
function computevar_term2!(
        ::Enable,
        buf::Vector{T}, # buffer.
        kq::Vector{T},
        C,
    )::T where {T}

    ldiv!(buf, C, kq) # overwrites buf.
    return dot(kq, buf)
end

function computevar_term2!(::Disable, ::Vector{T}, args...)::T where {T}
    return convert(T, NaN)
end

# front end.
"""
    @kwdef struct QueryOptions{MT <: BinarySelection, VT <: BinarySelection}
        compute_mean::MT = Enable()
        compute_variance::VT = Enable()
    end

Specifies whether the predicitve mean and/or the predictive variance should be computed during querying.

Possible values for each are: `LazyGPR.Enabled()` and `LazyGPR.Disabled()`.
"""
@kwdef struct QueryOptions{MT <: BinarySelection, VT <: BinarySelection}
    compute_mean::MT = Enable()
    compute_variance::VT = Enable()
end


# caching ϕ map over training inputs X.

struct WarpmapCache{PT}
    ϕ_X::PT
end

const CacheInfo = Union{Nothing, WarpmapCache}


# interpolation config.
abstract type ITPConfig end

# each local model has an adjustment map, s_x.

"""
    struct AdjustmentMap{T <: AbstractFloat}
        a::T
        b::T
        L::Int
    end

Container type for the observation variance adjustment map.
"""
struct AdjustmentMap{T <: AbstractFloat}
    a::T
    b::T
    L::Int
end


function AdjustmentMap(x0::T, y0::T, b_in, L::Integer)::AdjustmentMap{T} where {T}
    @assert y0 > 1
    @assert -b_in < x0 < b_in
    b = convert(T, b_in)
    b_L = b^L

    x0_L = x0^L
    a = -(b_L + y0 * (x0_L - b_L)) / x0_L
    if a < 0
        println("Warning: computed a negative value for the :a field of AdjustmentMap.")
    end

    return AdjustmentMap(a, b, L)
end

"""
    evalsmap(x, s::AdjustmentMap{T}) where T

Evaluates the ξ curve from the adjustment map container `s`, at distance `x`.
"""
function evalsmap(x, s::AdjustmentMap{T}) where {T}
    a, b, L = s.a, s.b, s.L
    if abs(x) >= b
        return convert(T, Inf)
    end

    x_L = x^L
    b_L = b^L
    return -(a * x_L + b_L) / (x_L - b_L)
end

# for illustration and diagnostics.
"""
    function evalsmap(u, x, s::AdjustmentMap)

Evaluates:
```
evalsmap(norm(u-x), s)
```
"""
function evalsmap(u, x, s::AdjustmentMap)
    return evalsmap(norm(u - x), s)
end


##### neighbourhood search for grid-based data.
struct StoreDist end

###### training

abstract type NeighbourhoodBuffer end
struct SKNeighbourhoodBuffer{T} <: NeighbourhoodBuffer
    U::Matrix{T}
    Xc::Matrix{T} # should stay constant during hoptim.
    RqX::Vector{T} # this is used in `evalsmap`, so it is used both during hyperparam optim and querying.

    y::Vector{T} # for the local neighbourhood. # should stay constant during hoptim.

    # # LOO-CV
    y_buf::Vector{T} # for K_inv * y # could reuse y at the cost of limiting future expansion that might require y to not be overwritten.
    # K_inv::Matrix{T} # reuse U.
end

struct DEKNeighbourhoodBuffer{T} <: NeighbourhoodBuffer
    U::Matrix{T}
    Xc::Matrix{T} # Xc[1:D, :] == Xa[1:D,:], and should stay constant during hoptim.
    ϕ_X::Vector{T} # should stay constant during hoptim.
    RqX::Vector{T} # this is used in `evalsmap`, so it is used both during hyperparam optim and querying.

    y::Vector{T} # for the local neighbourhood. # should stay constant during hoptim.

    # # LOO-CV
    y_buf::Vector{T} # for K_inv * y # could reuse y at the cost of limiting future expansion that might require y to not be overwritten.
    # K_inv::Matrix{T} # reuse U.
end

# use our own data structure in case we change the knn-inrange library in the future.
# The interface exposed to the public would not be tied to NN.
struct NNTree{T}
    tree::T
end

# for use with `computebuffer`.
struct SpatialSearchContainer{AV <: AbstractVector, NT <: NNTree}
    X::Vector{AV}
    nnt::NT
end


# data.

"""
    struct GPData{T, D, XT <: Union{Vector, SpatialSearchContainer, NTuple}}
        σ²::T
        inputs::XT
        outputs::Array{T,D}
    end

Container type for a conventional GPR model.
"""
struct GPData{T, D, XT <: Union{Vector, SpatialSearchContainer, NTuple}}
    σ²::T
    inputs::XT
    outputs::Array{T, D}
end

"""
    struct LazyGP{T, ST <: Union{Nothing, AdjustmentMap{T}}, XT, D}
        b_x::T
        s_map::ST
        data::GPData{T, D, XT}
    end

Container type for a the lazy-evaluation GPR model.

- `b_x` is the local neighborhood radius.
- `s_map` is the adjustment map for the observation noise model.
- `data` contains the GPR data.
"""
struct LazyGP{T, ST <: Union{Nothing, AdjustmentMap{T}}, XT, D}
    b_x::T
    s_map::ST
    data::GPData{T, D, XT}
end

function getdata(model::LazyGP)
    return model.data
end


#### traits.
abstract type ExtensionPkgs end


### traits for Inferring hyperparameters
abstract type ModelSelection end

"""
    struct MarginalLikelihood
This is a trait data type that indicates the marginal likelihood should be used.
"""
struct MarginalLikelihood <: ModelSelection end

"""
    struct LOOCV
This is a trait data type that indicates the leave-one-out cross-validation should be used. See Gaussian Process for Machine Learning (Rasmussen, Williams, 2005, DOI: 10.7551/mitpress/3206.001.0001) for more details.
"""
struct LOOCV <: ModelSelection end
