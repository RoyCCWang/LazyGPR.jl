####### specifying querying range on a grid

struct BoundingBox{N,T}
    top_left::NTuple{N,T}
    bottom_right::NTuple{N,T}
end

function getqueryranges(
    magnification_factor::Real,
    b_box::BoundingBox,
    Δts::Vector{T},
    ) where T <: AbstractFloat
    
    Δxs = collect( Δt/magnification_factor for Δt in Δts )

    top_left, bottom_right = b_box.top_left, b_box.bottom_right

    return collect(
        #convert(T, top_left[d]):convert(T, Δxs[d]):convert(T, bottom_right[d])
        range(
            convert(T, top_left[d]),
            step = convert(T, Δxs[d]),
            stop = convert(T, bottom_right[d]),
        )
        for d in eachindex(Δxs)
    ) # explicit convert to make sure the output type is of type T.
end

function step2linrange(::Type{T}, x_ranges::Vector{RT}) where {T, RT <: StepRangeLen}
    return collect( convert(LinRange{T,Int}, r) for r in x_ranges )
end