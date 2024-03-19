abstract type DEKernelFamily <: PositiveDefiniteKernel end

"""
    struct DEKernel{T <: AbstractFloat, WT, KT <: StationaryKernel} <: DEKernelFamily
        canonical::KT
        warpmap::WT # this is a callable function.
        κ::T
    end

The dimensional expansion (DE) kernel container data type.
- `canonical` is the canonical kernel. It should be a stationary kernel.
- `warpmap` is the warp function before the application of the gain multiplier.
- `κ` is the gain. Must be non-negative.

If `x`, `z` are inputs, then the kernel evaluates to:
```
evalkernel(
    canonical,
    norm( [x; κ*warpmap(x)] - [z; κ*warpmap(z)] ),
)
```
"""
struct DEKernel{T <: AbstractFloat, WT, KT <: StationaryKernel} <: DEKernelFamily
    canonical::KT
    warpmap::WT # this is a callable function.
    κ::T
end

# for generating a wrapper for warpmap, to use evalwarpmap().
function DEKernel(::Type{T}, warpmap) where T <: AbstractFloat
    return DEKernel(SqExpKernel(one(T)), warpmap, zero(T))
end

function createkernel(dek_old::DEKernel, θ::StationaryKernel, κ::AbstractFloat)
    return DEKernel(θ, dek_old.warpmap, κ)
end

##### for creating the cache ϕ_X.

"""
    computecachevars(::StationaryKernel, args...)
"""
function computecachevars(::StationaryKernel, args...)
    return nothing
end

"""
    computecachevars(dek::DEKernel, model::LazyGP)

Computes and stores the warpmap evaluations at each training input in `model`.
"""
function computecachevars(dek::DEKernel, model::LazyGP)
    return computecachevars(dek, getdata(model))
end

"""
    computecachevars(dek::DEKernel, data::GPData)
"""
function computecachevars(dek::DEKernel, data::GPData)
    ϕ_X = evalwarpmapbatch(dek, data.inputs)
    return WarpmapCache(ϕ_X)
end


# without warpmap. Use during hyperparams optim. Won't need to transport the warpmap to different processes.
struct DEKernelEcon{T <: AbstractFloat, KT <: StationaryKernel} <: DEKernelFamily
    canonical::KT
    κ::T
end


function createkernel(::DEKernelEcon, θ::StationaryKernel, κ::AbstractFloat)
    return DEKernelEcon(θ, κ)
end

function DEKernelEcon(dek::DEKernel)
    return DEKernelEcon(dek.canonical, dek.κ)
end

function unpackparams(p::AbstractVector, offset::Integer, dek::DEKernelFamily)
    M = getNparams(dek.canonical)
    return p[begin+offset:begin+offset+M-1], # for the canonical kernel.
    p[begin+offset+M], # κ
    offset+M + 1
end

function geteconkernel(a::Union{StationaryKernel, DEKernelEcon})
    return a
end

function geteconkernel(a::DEKernel)
    return DEKernelEcon(a)
end

# computer [X; phi_X'].
# function computeXc(
#     dek::DEKernel,
#     X::Matrix{T},
#     ) where T
    
#     Xc = zeros(T, size(X,1)+1, size(X,2))
#     computeXc!(Xc, dek, X)
    
#     return Xc # output.
# end

function getmetric(dek::DEKernelFamily)
    return getmetric(dek.canonical)
end

function evalwarpmapbatch(dek::DEKernel, X::AbstractVector)
    #return evalwarpmapbatch(dek.warpmap, X)
    return collect(
        dek.warpmap(x)
        for x in X
    )
end

function evalwarpmapbatch(
    dek::DEKernel, xrs::NTuple{D, AR},
    ) where {D, AR <: AbstractRange}

    #return evalwarpmapbatch(dek.warpmap, X)
    return collect(
        dek.warpmap(x)
        for x in Iterators.product(xrs...)
    )
end

"""
    evalkernel(x::Real, dek::DEKernelFamily)
Evaluates the canonical kernel in `dek` with the scalar input `x`. Note that DE kernels can only be evaluated for vector inputs.
"""
function evalkernel(x::Real, dek::DEKernelFamily)
    return evalkernel(x, dek.canonical)
end

"""
    evalde(dek::DEKernel{T}, x)::T where T <: AbstractFloat

Applies the gain κ.

Evaluates:
```
dek.κ*dek.warpmap(x)
```
"""
function evalde(dek::DEKernel{T}, x)::T where T <: AbstractFloat
    return dek.κ*dek.warpmap(x)
end

"""
    evalde(dek::DEKernel{T}, x)::T where T <: AbstractFloat

Does not apply the gain κ.

Evaluates:
```
dek.warpmap(x)
```
"""
function evalwarpmap(dek::DEKernel{T}, x)::T where T <: AbstractFloat
    return dek.warpmap(x)
end

### distributed computing.

# put relevant data on the workers.
function setup_evalwarpmap_dc(worker_list, dek::DEKernel)

    map(
        fetch,
        [
            DData.save_at(
                w, :dek_wk, dek,
            ) for w in worker_list
        ]
    )
    return nothing
end

function evalwarpmap_dc(Xq::Vector, worker_list::Vector{Int})

    # put parts of the query inputs on the workers.
    DData.scatter_array(
        :Xq_wk, Xq, worker_list,
    )

    tmp = map(
        fetch,
        [
            DData.get_from(
                w,
                :(
                    LazyGPR.evalwarpmapbatch(dek_wk, Xq_wk)
                ),
            ) for w in worker_list
        ]
    )
    out = collect( Iterators.flatten(tmp) )
    return out
end

function free_evalwarpmap_dc(worker_list)

    for w in worker_list
        DData.remove_from(w, :Xq_wk)
        DData.remove_from(w, :dek_wk)
    end

    return nothing
end