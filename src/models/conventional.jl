abstract type RcBuffer end

struct RcBufferDE{T <: Real} <: RcBuffer
    Rc::Matrix{T}
    Xc_mat::Matrix{T} # last row mutates., the rest are constants.
    w_X::Vector{T}

    #y_buf::Vector{T}
end

struct RcBufferStationary{T <: Real} <: RcBuffer
    Rc::Matrix{T} # constant.
    Xc_mat::Matrix{T} # constant.

    #y_buf::Vector{T}
end

function updateRc!(::StationaryKernel, B::RcBufferStationary)
    return nothing
end

function updateRc!(dek::DEKernelFamily, B::RcBufferDE)
    Xc_mat, w_X = B.Xc_mat, B.w_X

    # update with dek.κ
    for n in axes(Xc_mat, 2)
        Xc_mat[end, n] = dek.κ * w_X[n]
    end

    Distances.pairwise!(
        getmetric(dek), B.Rc, Xc_mat, dims = 2,
    )

    return nothing
end

####

"""
    fitGP!(
        U::Matrix{T}, # mutates.
        buffer::RcBuffer, # mutates.
        y::Vector{T},
        σ²::T,
        θ::PositiveDefiniteKernel,
    ) where T

Fit a conventional GPR model.
- `U`: kernel matrix buffer. Mutates.
- `buffer`: pair-wise distance buffer. Mutates.
- `y`: training outputs.
- `σ²`: observation noise variance.
- `θ`: the covariance function.

Returns:
- `c`: the cached coefficients for querying any predictive mean.
- `C`: the Cholesky factor for `U = K + σ²I`, where `K` is the kernel matrix.
"""
function fitGP!(
        U::Matrix{T}, # mutates.
        buffer::RcBuffer, # mutates.
        y::Vector{T},
        σ²::T,
        θ::PositiveDefiniteKernel,
    ) where {T}

    updateRc!(θ, buffer)

    updatekernelmatrix!(U, buffer.Rc, θ)
    for i in axes(U, 1)
        U[i, i] += σ²
    end

    #
    C = cholesky(U)
    c = C \ (vec(y))

    return c, C
end

function computenlli!(
        ms_trait::ModelSelection,
        U::Matrix{T},
        buffer::RcBuffer,
        y::Vector{T},
        σ²::T,
        θ::PositiveDefiniteKernel,
    ) where {T <: AbstractFloat}

    c, C = fitGP!(U, buffer, y, σ², θ)

    return evalHOcost!(ms_trait, C, c, y)
    #return dot(y, c) + logdet(C)
end


function hyperparametercost!(
        ms_trait::ModelSelection,
        U::Matrix, # mutates. buffer.
        buffer::RcBuffer,
        p::AbstractVector,
        unpackfunc,
        y::Vector,
        σ²,
        dek_ref::DEKernelFamily,
    )

    kernel_params, κ, _ = unpackfunc(p)

    dek = createkernel(
        dek_ref,
        createkernel(dek_ref.canonical, kernel_params...),
        κ,
    )

    return computenlli!(ms_trait, U, buffer, y, σ², dek)
end

function hyperparametercost!(
        ms_trait::ModelSelection,
        U::Matrix, # mutates. buffer.
        buffer::RcBuffer,
        p::AbstractVector,
        unpackfunc,
        y::Vector,
        σ²,
        θ_ref::StationaryKernel,
    )

    kernel_params, _ = unpackfunc(p)
    θ = createkernel(θ_ref, kernel_params...)

    return computenlli!(ms_trait, U, buffer, y, σ², θ)
end

# front end.
function create_hoptim_cost(
        ms_trait::ModelSelection,
        model::GPData,
        θ_ref::PositiveDefiniteKernel,
        args...
    )

    return setuphoptimcost(
            ms_trait, model.inputs, model.outputs, model.σ², θ_ref,
        ), nothing # more than one return objects to match the return behavior of create_hoptim_cost().
end

function setuphoptimcost(
        ms_trait::ModelSelection,
        X::Vector{AV},
        y::Vector{T},
        σ²::T,
        θ_ref::Union{StationaryKernel, DEKernel},
    ) where {T <: Real, AV <: AbstractVector{T}}

    N = length(y)
    @assert N == length(X)

    unpackfunc = pp -> unpackparams(pp, 0, θ_ref)
    U = zeros(T, N, N)

    buffer = setupRcbuffer(θ_ref, X)

    # the econ version doesn't store warpmap, which should make it faster to transfer to other processes during distributed computing.
    θ_ref_econ = geteconkernel(θ_ref)

    costfunc = pp -> hyperparametercost!(
        ms_trait, U, buffer, pp, unpackfunc, y, σ², θ_ref_econ,
    )
    return costfunc
end

function setupRcbuffer(θ::StationaryKernel, X::Vector)

    X_mat = vecs2mat(X)
    Rc = Distances.pairwise(getmetric(θ), X_mat)

    return RcBufferStationary(Rc, X_mat) #, zeros(T, length(X)))
end

function setupRcbuffer(dek::DEKernel, X)

    X_mat = vecs2mat(X)
    Rc = Distances.pairwise(getmetric(dek), X_mat)

    w_X = collect(evalwarpmap(dek, x) for x in X)

    return RcBufferDE(Rc, [X_mat; w_X'], w_X) #, similar(w_X))
end

# fits σ²
function setuphoptimcost1(
        ms_trait::ModelSelection,
        X::Vector{AV},
        y::Vector{T},
        θ_ref::Union{StationaryKernel, DEKernel},
    ) where {T <: Real, AV <: AbstractVector{T}}

    N = length(y)
    @assert N == length(X)

    unpackfunc = pp -> unpackparams(pp, 0, θ_ref)
    U = zeros(T, N, N)

    buffer = setupRcbuffer(θ_ref, X)

    # the econ version doesn't store warpmap, which should make it faster to transfer to other processes during distributed computing.
    θ_ref_econ = geteconkernel(θ_ref)

    costfunc = pp -> hyperparametercost!(
        ms_trait, U, buffer, view(pp, 1:(length(pp) - 1)), unpackfunc, y, pp[end], θ_ref_econ,
    )
    return costfunc
end

function setuphoptimcost2(
        ms_trait::ModelSelection,
        X::Vector{AV},
        y::Vector{T},
        θ_ref::Union{StationaryKernel, DEKernel},
    ) where {T <: Real, AV <: AbstractVector{T}}

    N = length(y)
    @assert N == length(X)

    unpackfunc = pp -> unpackparams(pp, 0, θ_ref)
    U = zeros(T, N, N)

    buffer = setupRcbuffer(θ_ref, X)

    # the econ version doesn't store warpmap, which should make it faster to transfer to other processes during distributed computing.
    θ_ref_econ = geteconkernel(θ_ref)

    costfunc = pp -> hyperparametercost!(
        ms_trait, U, buffer, view(pp, 1:(length(pp) - 1)), unpackfunc, y, pp[end], θ_ref_econ,
    )
    return costfunc
end

###### query

struct DenseGPModel{T <: Real, CT}
    c::Vector{T}
    C::CT
    Xc_mat::Matrix{T}
end

function getXcmat(dek::DEKernel, X::Vector)

    X_mat = vecs2mat(X)
    phi_X = collect(evalde(dek, x) for x in X)

    return [X_mat; phi_X']
end

function getXcmat(::StationaryKernel, X::Vector)
    return vecs2mat(X)
end

"""
    fitGP(X::Vector, y::Vector{T}, σ²::AbstractFloat, θ) where T <: AbstractFloat


Fit a conventional GPR model.
- `U`: kernel matrix buffer. Mutates.
- `buffer`: pair-wise distance buffer. Mutates.
- `y`: training outputs.
- `σ²`: observation noise variance.
- `θ`: the covariance function.

Returns an object of type `DenseGPModel`.
"""
function fitGP(X::Vector, y::Vector{T}, σ²::AbstractFloat, θ) where {T <: AbstractFloat}

    buffer = setupRcbuffer(θ, X)
    N = length(X)
    U = zeros(T, N, N)

    c, C = fitGP!(U, buffer, y, σ², θ)

    return DenseGPModel(c, C, getXcmat(θ, X))
end

"""
    queryGP(xrs::NTuple{D, AR}, θ, model::DenseGPModel) where {D, AR <: AbstractRange}

Query a conventional GPR at input `xrs`.
"""
function queryGP(xrs::NTuple{D, AR}, θ, model::DenseGPModel) where {D, AR <: AbstractRange}

    return collect(
        queryGP(collect(x), θ, model)
            for x in Iterators.product(xrs...)
    )
end

"""
    queryGP(
        xq::AbstractVector{T},
        kernel::Union{DEKernel, StationaryKernel},
        model::DenseGPModel{T},
    ) where T <: AbstractFloat
"""
function queryGP(
        xq::AbstractVector{T},
        kernel::Union{DEKernel, StationaryKernel},
        model::DenseGPModel{T},
    ) where {T <: AbstractFloat}

    return queryGP!(Memory{T}(undef, length(model.c)), xq, kernel, model)
end

function queryGP!(
        buf::AbstractVector{T},
        xq::AbstractVector{T},
        dek::DEKernel,
        model::DenseGPModel{T},
    ) where {T <: AbstractFloat}

    c, C, Xc_mat = model.c, model.C, model.Xc_mat

    xcq_mat = reshape(
        getxc(xq, dek),
        (length(xq) + 1, 1),
    )

    Rc = Distances.pairwise(
        getmetric(dek),
        Xc_mat,
        xcq_mat,
        dims = 2,
    )
    kq = vec(collect(evalkernel(r, dek.canonical) for r in Rc))

    pred_mean = dot(kq, c)

    ldiv!(buf, C, kq)
    pred_var = evalkernel(zero(T), dek.canonical) - dot(kq, buf)

    return pred_mean, pred_var
end

function queryGP!(
        buf::AbstractVector{T},
        xq::AbstractVector{T},
        θ::StationaryKernel,
        model::DenseGPModel{T}
    ) where {T <: AbstractFloat}

    c, C, X_mat = model.c, model.C, model.Xc_mat

    Rc = Distances.pairwise(
        getmetric(θ),
        X_mat,
        reshape(xq, length(xq), 1),
        dims = 2,
    )
    kq = vec(collect(evalkernel(r, θ) for r in Rc))

    pred_mean = dot(kq, c)

    ldiv!(buf, C, kq)
    pred_var = evalkernel(zero(T), θ) - dot(kq, buf)

    return pred_mean, pred_var
end

# identical to queryGP!, without computing the variance.
function querymean(
        xq::AbstractVector{T},
        dek::DEKernel,
        model::DenseGPModel{T},
    ) where {T <: AbstractFloat}

    c, Xc_mat = model.c, model.Xc_mat

    xcq_mat = reshape(
        getxc(xq, dek),
        (length(xq) + 1, 1),
    )

    Rc = Distances.pairwise(
        getmetric(dek),
        Xc_mat,
        xcq_mat,
        dims = 2,
    )
    kq = vec(collect(evalkernel(r, dek.canonical) for r in Rc))

    pred_mean = dot(kq, c)

    return pred_mean
end

# identical to queryGP!, without computing the variance.
function querymean(
        xq::AbstractVector{T},
        θ::StationaryKernel,
        model::DenseGPModel{T}
    ) where {T <: AbstractFloat}

    c, X_mat = model.c, model.Xc_mat

    Rc = Distances.pairwise(
        getmetric(θ),
        X_mat,
        reshape(xq, length(xq), 1),
        dims = 2,
    )
    kq = vec(collect(evalkernel(r, θ) for r in Rc))

    pred_mean = dot(kq, c)

    return pred_mean
end


function queryGPvariance!(
        buf::AbstractVector{T},
        xq::AbstractVector{T},
        dek::DEKernel,
        model::DenseGPModel{T},
    ) where {T <: AbstractFloat}

    c, C, Xc_mat = model.c, model.C, model.Xc_mat

    xcq_mat = reshape(
        getxc(xq, dek),
        (length(xq) + 1, 1),
    )

    Rc = Distances.pairwise(
        getmetric(dek),
        Xc_mat,
        xcq_mat,
        dims = 2,
    )
    kq = vec(collect(evalkernel(r, dek.canonical) for r in Rc))

    ldiv!(buf, C, kq)
    pred_var = evalkernel(zero(T), dek.canonical) - dot(kq, buf)

    return pred_var
end

function queryGPvariance!(
        buf::AbstractVector{T},
        xq::AbstractVector{T},
        θ::StationaryKernel,
        model::DenseGPModel{T}
    ) where {T <: AbstractFloat}

    c, C, X_mat = model.c, model.C, model.Xc_mat

    Rc = Distances.pairwise(
        getmetric(θ),
        X_mat,
        reshape(xq, length(xq), 1),
        dims = 2,
    )
    kq = vec(collect(evalkernel(r, θ) for r in Rc))

    ldiv!(buf, C, kq)
    pred_var = evalkernel(zero(T), θ) - dot(kq, buf)

    return pred_var
end
