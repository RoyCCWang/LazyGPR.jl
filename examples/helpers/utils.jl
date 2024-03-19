#tensor product filtering for dimension D discrete signal.
function ranges2collection(
    x_ranges::Vector{LinRange{T,L}},
    ::Val{D},
    )::Array{Vector{T},D} where {T,D,L}

    #set up.
    @assert !isempty(x_ranges)
    @assert length(x_ranges) == D
    N_array = collect( length(x_ranges[d]) for d = 1:D )
    sz_N = tuple(N_array...)

    #Position.
    X_nD = Array{Vector{T},D}(undef,sz_N)
    for 𝑖 in CartesianIndices(sz_N)
        X_nD[𝑖] = Vector{T}(undef,D)

        for d = 1:D
            X_nD[𝑖][d] = x_ranges[d][𝑖[d]]
        end
    end

    return X_nD
end

#converts compact domain x ∈ [a,b] to compact domain out ∈ [c,d].
function convertcompactdomain(x::T, a::T, b::T, c::T, d::T)::T where T <: Real

    return (x-a)*(d-c)/(b-a)+c
end

function convertcompactdomain(x::Vector{T}, a::Vector{T}, b::Vector{T}, c::Vector{T}, d::Vector{T})::Vector{T} where T <: Real

    return collect( convertcompactdomain(x[i], a[i], b[i], c[i], d[i]) for i = 1:length(x) )
end
