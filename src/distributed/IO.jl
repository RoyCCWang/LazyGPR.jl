


function subscript2filename(x::Tuple)::String

    s = replace("$x", "," => "_")
    s = replace(s, "(" => "")
    s = replace(s, ")" => "")
    s = replace(s, " " => "")
    
    return s
end

function filename2subscript(f::String)

    v = collect(
        string(s)
        for s in split(f,"_")
    )
    return tuple(v...) # type unstable.
end


#### file IO.


# function loadresults(
#     save_folder::String,
#     info::Union{GridBatches, IntervalCover},
#     )

#     itr = getitr(info)

#     return collect(
#         deserialize(
#             joinpath(
#                 save_folder,
#                 subscript2filename(s),
#             ),
#         )
#         for s in itr
#     )
# end

function assemblequery(
    load_dir::String,
    batch_info::GridBatches,
    )

    data = loadresults(load_dir, batch_info)
    return assemblequery(data, batch_info)
end

function assemblequery(
    data::Array{Array{T,D}, D},
    batch_info::GridBatches,
    ) where {T, D}

    itr = getitr(batch_info)
    @assert size(data) == size(itr)

    # query_grids = collect(
    #     extractranges(s, batch_info)[begin]
    #     for s in itr
    # )
    
    query_ranges = collect(
        extractranges(s, batch_info)[begin+1]
        for s in itr
    )

    # assemble output.
    sz_vec = collect(
        x[end]
        for x in query_ranges[end]
    )
    Y = Array{T,D}(undef, sz_vec...)
    for s in itr
        Y[query_ranges[s...]...] = data[s...]
    end

    return Y
end


# function coord2inds(cr::AR, N::Integer) where AR <: AbstractRange
#     #
#     return range(1, N, length = length(cr))
# end

# function coord2inds(crs::Vector{AR}, Ns::Vector{Int}) where AR <: AbstractRange
#     @assert length(crs) == length(Ns)

#     return collect( coord2inds(crs[d], Ns[d]) for d in eachindex(crs))
# end
