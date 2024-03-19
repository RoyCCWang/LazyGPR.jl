
abstract type BatchInfo end

struct GridBatches{RT <: AbstractRange} <: BatchInfo
    X_ranges::Vector{RT} # length D.
    batch_sizes::Vector{Int} # length D.
end

function extractranges(s::Tuple, info::GridBatches)

    return extractranges(s, info.X_ranges, info.batch_sizes)
end

function generatebatches(info::GridBatches)
    return generatebatches(info.X_ranges, info.batch_sizes)
end

function getitr(info::GridBatches)
    N_batches = generatebatches(info)
    return Iterators.product(( 1:Nd for Nd in N_batches )...)
end

function getqueryitr(s, info::GridBatches)

    local_x_ranges, _ = extractranges(s, info.X_ranges, info.batch_sizes)
    return Iterators.product(local_x_ranges...)
end

function selectrangeinclusive(itr, a::Integer, b::Integer)
    return Iterators.drop(Iterators.take(itr, b), a-1)
end


function generatebatches(
    x_ranges::Vector{RT},
    batch_sizes::Vector{Int},
    ) where RT <: AbstractRange

    @assert length(x_ranges) == length(batch_sizes)

    #
    N_batches = collect(
        cld(length(x_ranges[d]), batch_sizes[d])
        for d in eachindex(x_ranges)
    )

    return N_batches
end


function extractranges(
    s::Tuple, # a multi-index from Iterators.product.
    x_ranges::Vector{RT},
    batch_sizes::Vector{Int},
    ) where RT <: AbstractRange

    @assert length(batch_sizes) == length(x_ranges) == length(s)

    out = similar(x_ranges)
    range_bounds = Vector{UnitRange{Int}}(undef, length(x_ranges))
    for d in eachindex(x_ranges)
        
        N = batch_sizes[d]

        st_ind = (s[d]-1)*N + 1
        fin_ind = st_ind + N -1
        fin_ind = min(fin_ind, length(x_ranges[d]))

        out[d] = x_ranges[d][begin+st_ind-1:begin+fin_ind-1]
        range_bounds[d] = st_ind:fin_ind
    end

    return out, range_bounds
end



