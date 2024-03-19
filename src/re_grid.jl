
##### scatter to grid.


#### re-grid.

function scatter2itp_dc(
    alg::ExtensionPkgs,
    Xrs::NTuple{D, AR},
    dek::DEKernel,
    X::Vector, y::Vector{T},
    rel_err_ub, max_iters::Integer,
    ) where {T <: Real, D, AR <: AbstractRange}
    
    if minimum(length.(Xrs)) < 4
        error("Please specify a grid that has more than 4 entries in the smallest dimension.")
    end

    if 2^max_iters >= minimum(size(A0))
        max_iters = floor(Int, log2(minimum(size(A0)))) -1
    end


    # prepare grid inputs.
    X_grid = collect( Iterators.product(Xrs...) ) # there might be a better way than allocating here.
    
    # distributed computing to get A0.
    setup_evalwarpmap_dc(worker_list, dek)

    A0 = reshape(
        evalwarpmap_dc(vec(X_grid), worker_list),
        size(X_grid),
    )
    
    free_evalwarpmap_dc(worker_list)

    return scatter2itp(
        alg, Xrs, A0, X, y, rel_err_ub, max_iters,
    )
end

# f: NTuple |-> real.
function scatter2itp(
    alg::ExtensionPkgs,
    Xrs::NTuple{D, AR},
    dek::DEKernel,
    X::Vector, y::Vector{T},
    rel_err_ub, max_iters::Integer,
    #worker_list = nworkers(),
    ) where {T <: Real, D, AR <: AbstractRange}
    
    if minimum(length.(Xrs)) < 4
        error("Please specify a grid that has more than 4 entries in the smallest dimension.")
    end

    A0 = collect(
        evalwarpmap(dek, x) for x in Iterators.product(Xrs...)
    )

    if 2^max_iters >= minimum(size(A0))
        max_iters = floor(Int, log2(minimum(size(A0)))) -1
    end

    return scatter2itp(
        alg, Xrs, A0, X, y, rel_err_ub, max_iters,
    )
end

function scatter2itp(
    alg::ExtensionPkgs,
    Xrs::NTuple{D, AR},
    A0::AbstractArray{T},
    X::Vector, y::Vector{T},
    rel_err_ub, max_iters::Integer,
    ) where {T <: Real, D, AR <: AbstractRange}

    #setup_evalwarpmap_dc(worker_list, dek)

    if minimum(length.(Xrs)) < 4
        error("Please specify a grid that has more than 4 entries in the smallest dimension.")
    end

    if 2^max_iters >= minimum(size(A0))
        max_iters = floor(Int, log2(minimum(size(A0)))) -1
    end
    
    for m = 1:max_iters

        # re-grid.
        M = 2^(m-1)
        f_X, Xrs_m = get_downsampled_array(A0, Xrs, M)

        warpmap = create_warp_map(alg, Xrs_m, f_X)

        # refine.
        warpmap_U = warpmap.(X)
        rel_err = norm(warpmap_U-y)/norm(y)

        if rel_err <= rel_err_ub
            return warpmap, rel_err, m
        end

        if m == max_iters
            println("Maximum iters reached in scatter2itp(). Return latest result.")
            return warpmap, rel_err, m
        end
    end

    return nothing, convert(T, Inf) # should never happen.
end

function get_downsampled_array(A::Array{T,2}, M::Integer) where T
    return A[begin:M:end,begin:M:end]
end

function get_downsampled_array(A::Array{T,2}, Xrs, M::Integer) where T
    return A[begin:M:end,begin:M:end], ntuple(d->Xrs[d][begin:M:end], 2)
end

function get_downsampled_array(A::Array{T,3}, M::Integer) where T
    return A[begin:M:end, begin:M:end, begin:M:end]
end

function get_downsampled_array(A::Array{T,3}, Xrs, M::Integer) where T
    return A[begin:M:end, begin:M:end, begin:M:end], ntuple(d->Xrs[d][begin:M:end], 3)
end