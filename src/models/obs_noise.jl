
##### Noise model

# function applysmap!(::Nothing, args...)
#     return nothing
# end

function applynoisemodel!(dek::DEKernelFamily, args...)
    return applynoisemodel!(dek.canonical, args...)
end

function applynoisemodel!(
    ::SqDistanceKernel,
    s::AdjustmentMap, # dispatch.
    U::Matrix, # mutates. output.
    σ²,
    sq_RqX,
    )
    @assert length(sq_RqX) == size(U,1) == size(U,2)
    
    for i in eachindex(sq_RqX)
        U[i,i] += σ² + evalsmap(sqrt(sq_RqX[i]), s)  - 1
    end

    return processdiagonals(U)
end

# returns a principle submatrix of U that has finite diagonal entries.
# returns `valids` if there are non-finite diagonal entries. `Nothing`  if all diagonal entries are finite.
function applynoisemodel!(
    ::DistanceKernel,
    s::AdjustmentMap, # dispatch.
    U::Matrix, # mutates. output.
    σ²,
    R_xq_X,
    )
    @assert length(R_xq_X) == size(U,1) == size(U,2)
    
    for i in eachindex(R_xq_X)
        U[i,i] += σ² + evalsmap(R_xq_X[i], s) - 1
    end

    return processdiagonals(U)
end

function processdiagonals(U::Matrix)
    #
    # store `false` if the i-th row & column has a non-finite value.
    valids = trues(size(U,1))

    for i in axes(U,1)
        
        if !isfinite(U[i,i])
            valids[i] = false
        end
    end
    
    if all(valids)
        # all entries are finite.
        return U, nothing
    end

    # remove the rows and columns that have a non-finite diagonal entry.
    #V = @view U[valids, valids]
    V = U[valids,valids] # subsequent operations benfits from a continuous matrix, so we don't return the view.
    return V, valids
end

# returns a principle submatrix of U that has finite diagonal entries.
# returns `valids` if there are non-finite diagonal entries. `Nothing`  if all diagonal entries are finite.
function applynoisemodel!(
    ::StationaryKernel,
    ::Nothing,
    U::Matrix, # mutates. output.
    σ²,
    args...
    )
    @assert size(U,1) == size(U,2)
    
    for i in eachindex(sq_RqX)
        U[i,i] += σ²
    end

    return U, nothing # don't check diagonals since s-map isn't used.
end