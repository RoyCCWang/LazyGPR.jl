
##### assemble the local training set.



# for GP on a grid. Reduce the grid neighbourhood to Euclidean ball with radius b_x and the distances in RqX.
function filterdata(
    ::StationaryKernel,
    ::Nothing,
    dist, x_ranges::Vector{AR}, RqX, b_x, y,
    ) where AR <: AbstractRange
    
    # filter out the points that are too, based on RqX and b_x.
    it = Iterators.product(x_ranges...)

    max_dist = getcomparedist(dist, b_x)
    vec_tuples = vec(
        collect(
            Iterators.filter(
                xx->xx[begin]<max_dist,
                zip(RqX, it, y),
            )
        )
    )
    
    # parse filtered result into separate variables.
    RqX_filtered = collect( i[begin] for i in vec_tuples )
    y_filtered = collect( i[begin+2] for i in vec_tuples )

    D = length(x_ranges)
    N = length(RqX_filtered)
    X_mat_filtered = reshape(
        collect(
            Iterators.flatten(
                i[begin+1] for i in vec_tuples
            )
        ),
        D, N,
    )

    return X_mat_filtered, RqX_filtered, y_filtered
end

# no cahching of glocal ϕ_X.
function filterdata(
    dek::DEKernel,
    ::Nothing,
    dist, x_ranges::Vector{AR}, RqX, b_x, y,
    ) where AR <: AbstractRange

    X_mat, RqX_filtered, y_filtered = filterdata(
        dek.canonical,
        nothing,
        dist, x_ranges, RqX, b_x, y,
    )
    #@show X_mat # debug
    Xc_mat = computeXc(dek, X_mat)
    #@show Xc_mat # debug

    return Xc_mat, RqX_filtered, y_filtered
end

# the version for supplied cache ϕ_X.
# for GP on a grid. Reduce the grid neighbourhood to Euclidean ball with radius b_x and the distances in RqX.
function filterdata(
    dek::DEKernel,
    ϕ_X::Array,
    dist, x_ranges::Vector{AR}, RqX, b_x, y,
    ) where AR <: AbstractRange
    
    #@show norm(ϕ_X) # debug

    # filter out the points that are too, based on RqX and b_x.
    it = Iterators.product(x_ranges...)

    max_dist = getcomparedist(dist, b_x)
    vec_tuples = vec(
        collect(
            Iterators.filter(
                xx->xx[begin]<max_dist,
                zip(RqX, it, y, ϕ_X),
            )
        )
    )
    
    # parse filtered result into separate variables.
    RqX_filtered = collect( i[begin] for i in vec_tuples )
    y_filtered = collect( i[begin+2] for i in vec_tuples )
    ϕ_X_filtered = collect( i[begin+3] for i in vec_tuples )
    #@show ϕ_X_filtered # debug

    D = length(x_ranges)
    N = length(RqX_filtered)
    X_mat = reshape(
        collect(
            Iterators.flatten(
                i[begin+1] for i in vec_tuples
            )
        ),
        D, N,
    )
    #@show X_mat # debug
    #mul!(X_mat, dek.sqrt_1_minux_κ, X_mat) # overwrite X_mat with X_mat_tilde, the scaled version.
    Xc_mat = [X_mat; ϕ_X_filtered']
    #@show Xc_mat # debug

    return Xc_mat, RqX_filtered, y_filtered
end

# no cache variables implemented for stationary kernels.
function assemblecachegrid(::StationaryKernel, args...)
    return nothing
end

function assemblecachegrid(::DEKernel, ::Nothing, args...)
    return nothing
end

function assemblecachegrid(
    ::DEKernel,
    cvars::WarpmapCache,
    ind_ranges,
    )
    #
    ϕ_X_grid = collect(
        cvars.ϕ_X[CartesianIndex(i)]
        for i in Iterators.product(ind_ranges...)
    )
    return ϕ_X_grid
end

# for GP on a grid.
function assembledata(
    image_ranges::Vector{AR},
    image_intensities::Array, # multi-dim array
    xq,
    b_x,
    θ::PositiveDefiniteKernel,
    cvars::CacheInfo,
    ) where AR <: AbstractRange
    
    dist = getmetric(θ)

    # extract the preliminary signal neighbourhood centered at xq.
    Xg_ranges, ind_ranges = findsubgrid(
        image_ranges, b_x, xq,
    )
    yg_nD = collect(
        image_intensities[CartesianIndex(i)]
        for i in Iterators.product(ind_ranges...)
    )
    ϕ_X_grid = assemblecachegrid(θ, cvars, ind_ranges)

    # refine the signal neighbourhood.
    R_xq_Xg = assembleRqX(Xg_ranges, xq, dist)

    Xc_mat, R_xq_X, y = filterdata(
        θ, ϕ_X_grid,
        dist, Xg_ranges, R_xq_Xg, b_x, vec(yg_nD),
    )

    return Xc_mat, R_xq_X, y
end

# test function. Ensure the training set is within b_x distance from xq.
function testassembledata(
    image_ranges, im_y, xq, b_x, dist, cvars,
    )

    _, R_xq_X, _ = assembledata(
        image_ranges, im_y, xq, b_x, dist, cvars,
    )

    max_dist = getcomparedist(dist, b_x)
    return maximum(R_xq_X) < max_dist
end


# find the subset of x_ranges that contains x within -+ b_x tolerance.
function findsubgrid(x_ranges::Vector{AR}, b_x, xq) where AR <: AbstractRange

    y_ranges = similar(x_ranges)
    #inds = Vector{Tuple{Int,Int}}(undef, length(x_ranges))
    ind_ranges = Vector{UnitRange{Int}}(undef, length(x_ranges))
    for d in eachindex(x_ranges)

        x_d = xq[d]
        rd = x_ranges[d]
        Δt = step(rd)
        t0 = rd[begin]

        r_lb = x_d - b_x
        r_ub = x_d + b_x
        
        # +1 since the formula is for 0-indexing n: t = t0 + n*Δt.
        n_lb = max(floor(Int, (r_lb - t0)/Δt) +1, 1)
        n_ub = min(ceil(Int, (r_ub - t0)/Δt) +1, length(rd))
        #@show d, n_lb, n_ub

        y_ranges[d] = rd[n_lb:n_ub]
        #inds[d] = (n_lb, n_ub)
        ind_ranges[d] = n_lb:n_ub
    end

    return y_ranges, ind_ranges
end

# test function. Verifies x_ranges contain xq +/- b_x.
function testfindsubgrid(x_ranges::Vector{AR}, xq, b_x) where AR <: AbstractRange
    @assert length(x_ranges) == length(xq)

    flag = true
    for d in eachindex(x_ranges)
        rd = x_ranges[d]

        flag1 = (rd[begin] <= xq[d] - b_x)
        
        flag2 = (rd[end] >= xq[d] + b_x)

        flag = flag && flag1 && flag2
    end
    return flag
end