

### elemntary Kernels.
# # All kernels are normalized to have maximum value of 1

abstract type PositiveDefiniteKernel end
abstract type StationaryKernel <: PositiveDefiniteKernel end

abstract type SqDistanceKernel <: StationaryKernel end
abstract type DistanceKernel <: StationaryKernel end

function getmetric(::DistanceKernel)
    return Distances.Euclidean()
end

function getmetric(::SqDistanceKernel)
    return Distances.SqEuclidean()
end


# # Compact spline kernels. See  Cor. 9.14, Wendland 2005.
# generalizes Spline12, 23, 34 kernels to arbitrary (D,q), where q is mean-square differentiability of the kernel.

abstract type DifferentiabilityTrait end

"""
    struct Order3
This is a trait data type that indicates the order is 3.
"""
struct Order3 <: DifferentiabilityTrait end

"""
    struct Order2
This is a trait data type that indicates the order is 2.
"""
struct Order2 <: DifferentiabilityTrait end

"""
    struct Order1
This is a trait data type that indicates the order is 1.
"""
struct Order1 <: DifferentiabilityTrait end


"""
    struct WendlandSplineKernel{T <: AbstractFloat, N} <: DistanceKernel
        a::T

        # buffer.
        l::Int
        c::NTuple{N,T}
        m::Int
    end

The Wendland spline kernels, which are compactly supported. See (Wendland 2004, DOI: 10.1017/CBO9780511617539) for more information.
- `a` controls the bandwidth. Must be larger than zero.
- `l`, `c`, `m` are internal intermediate objects that represent a Wendland spline of a specified order and dimension.
"""
struct WendlandSplineKernel{T <: AbstractFloat, N} <: DistanceKernel
    #a::T
    a::T

    # buffer.
    l::Int
    c::NTuple{N,T} # first index is the 0-th order term.

    # dispatch
    #differentiability::DT
    m::Int # l + order, l = getsplineparameter(D, order)
    #Z::Int # normalizing constant for the spline kernel to have a maximum of 1. It is the offset constant from Cor. 19.4 in the second factor.
end

function createkernel(θ::WendlandSplineKernel, a::AbstractFloat)
    return WendlandSplineKernel(a, θ.l, θ.c, θ.m)
end


# Cor. 9.14, Wendland 2005.
function getsplineparameter(D::Int, q::Int)::Int
    return div(D,2)+q+1 # note that div(D,2) == floot(Int, D/2)
end

"""
    WendlandSplineKernel(::Order3, a::T, D::Int)::WendlandSplineKernel{T} where T
Constructor function for an order 3, `D`-dimensional Wendland spline.
"""
function WendlandSplineKernel(::Order3, a::T, D::Int)::WendlandSplineKernel{T} where T
    # based on Cor. 9.14, Wendland.
    l = getsplineparameter(D, 3)
    c = tuple(
        one(T),
        convert(T, l + 3),
        convert(T, (6*l^2 + 36*l + 45)/15),
        convert(T, (l^3 + 9*l^2 + 23*l + 15)/15),
    )
    
    return WendlandSplineKernel(a, l, c, l+3)
end

"""
WendlandSplineKernel(::Order2, a::T, D::Int)::WendlandSplineKernel{T} where T
Constructor function for an order 2, `D`-dimensional Wendland spline.
"""
function WendlandSplineKernel(::Order2, a::T, D::Int)::WendlandSplineKernel{T} where T
    
    l = getsplineparameter(D, 2)
    c = tuple(
        one(T),
        convert(T, l + 2),
        convert(T, (l^2 + 4*l + 3)/3),
    )
    
    return WendlandSplineKernel(a, l, c, l+2)
end

"""
WendlandSplineKernel(::Order1, a::T, D::Int)::WendlandSplineKernel{T} where T
Constructor function for an order 1, `D`-dimensional Wendland spline.
"""
function WendlandSplineKernel(::Order1, a::T, D::Int)::WendlandSplineKernel{T} where T
    
    l = getsplineparameter(D, 1)
    c = tuple(
        one(T),
        convert(T, l + 1),
    )
    
    return WendlandSplineKernel(a, l, c, l+1)
end

"""
    evalkernel(τ::Real, θ::WendlandSplineKernel{T,N})::T where {T <: AbstractFloat, N}

Evaluates the Wendland spline kernel `θ`.
"""
function evalkernel(τ::Real, θ::WendlandSplineKernel{T,N})::T where {T <: AbstractFloat, N}

    r = τ*θ.a
    tmp = 1-r
    if signbit(tmp)
        return zero(T)
    end

    factor1 = tmp^θ.m
    #factor2 = (sum( c[i]*r^(i-1) for i in eachindex(c) ) ) # use Horner's method if speed it an issue.
    factor2 = evalpoly(r, θ.c) # use Horner's method if speed it an issue.
    return factor1*factor2
end

# # exponential kernel, aka Ornstein-Uhlenbeck covariance function.
struct ExpKernel{T <: AbstractFloat} <: DistanceKernel
    # k(t,z) = q/(2*λ) * exp(-λ*abs(t-z)).
    λ::T # larger than 0.
    q::T # larger than 0.
    c::T # q/(2*λ)
end

function ExpKernel(λ::T, q::T)::ExpKernel{T} where T
    return ExpKernel(λ, q, q/(2*λ))
end

function createkernel(::ExpKernel{T}, λ::T, q::T)::ExpKernel{T} where {T <: AbstractFloat}
    return ExpKernel(λ, q)
end

"""
    evalkernel(τ::Real, θ::ExpKernel{T})::T where T <: AbstractFloat

Evaluates the exponential (Ornstein–Uhlenbeck) kernel `θ`.
"""
function evalkernel(τ::Real, θ::ExpKernel{T})::T where T <: AbstractFloat
    return θ.c * exp(-θ.λ*τ)
end

# Matern kernel.
struct Matern3Halfs{T <: AbstractFloat} <: DistanceKernel
    # k(x,z) = b*(1 + a*norm(x-z))*exp(-a*norm(x-z))
    a::T # larger than 0.
    b::T # larger than 0.
end

function createkernel(::Matern3Halfs{T}, a::T, b::T)::Matern3Halfs{T} where {T <: AbstractFloat}
    return Matern3Halfs(a, b)
end

"""
    evalkernel(τ::Real, θ::Matern3Halfs{T})::T where T <: AbstractFloat

Evaluates the Matern 3/2 kernel `θ`.
"""
function evalkernel(τ::Real, θ::Matern3Halfs{T})::T where T <: AbstractFloat
    tmp = θ.a*τ
    return θ.b*(1+tmp) * exp(-tmp)
end


# # Stationary kernels with squared distance as input.

# ## Eq. 4.19, pg 86, GPML 2006.
struct RQKernel{T <: AbstractFloat, ET <: Real} <: SqDistanceKernel
    #a::T # 1/(2*n*l^2) from GPML 2006.
    a::T # larger than 0.
    n::ET # exponent.
end

function createkernel(::RQKernel{T,ET}, a::T, n::ET)::RQKernel{T,ET} where {T <: AbstractFloat, ET <: Real}
    return RQKernel(a, n)
end

"""
    evalkernel(τ_sq::Real, θ::RQKernel{T,ET})::T where {T <: AbstractFloat, ET}

Evaluates the rational quadratic kernel `θ`.
"""
function evalkernel(τ_sq::Real, θ::RQKernel{T,ET})::T where {T <: AbstractFloat, ET}
    #τ_sq = τ^2
    #return (1 + θ.a * τ_sq )^(θ.n)
    return (muladd(θ.a, τ_sq, 1))^(θ.n)
end

function getNparams(::Union{RQKernel, Matern3Halfs, ExpKernel})
    return 2
end

"""
struct SqExpKernel{T <: AbstractFloat} <: SqDistanceKernel
    a::T
end

The square exponential kernel. `a` must be larger than zero. If `τ` is the distance input, then the kernel output is:
```
exp(-a*τ)
```
"""
struct SqExpKernel{T <: AbstractFloat} <: SqDistanceKernel
    a::T # larger than 0.
end

function createkernel(::SqExpKernel{T}, a::T)::SqExpKernel{T} where T <: AbstractFloat
    return SqExpKernel(a)
end

"""
    evalkernel(τ_sq::Real, θ::SqExpKernel{T})::T where T <: AbstractFloat
Evaluates the square exponential kernel `θ`.
"""
function evalkernel(τ_sq::Real, θ::SqExpKernel{T})::T where T <: AbstractFloat
    #τ_sq = τ^2
    return exp(-θ.a*τ_sq)
end

function getNparams(::Union{WendlandSplineKernel, SqExpKernel})
    return 1
end

function unpackparams(p::AbstractVector, offset::Integer, θ::StationaryKernel)
    M = getNparams(θ)
    return p[begin+offset:begin+offset+M-1], offset+M
end