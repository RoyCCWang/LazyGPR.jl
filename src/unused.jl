
# equivalent to eval_negative_LOOCV()
function eval_LOOCV(
    K_inv::Matrix{T},
    K_inv_y::Vector{T},
    y::Vector{T},
    ) where T <: AbstractFloat

    score = zero(T)
    for i in eachindex(y)
        v_i = 1/K_inv[i,i]
        m_i = y[i] - v_i*K_inv_y[i]
        score += -log(v_i)/2 - (y[i]-m_i)^2/(2*v_i)
    end

    return score
end

# 2.5 order of magnitude faster.
# for grid-based signals, it is faster to get the rectangle that contains a sphere of radius b_x.
function computevariancerect(
    Xrs::NTuple{D, AR},
    A::Array{T,D},
    b_x::T,
    ) where {T <: AbstractFloat, D, AR <: AbstractRange}

    vs = zeros(T, size(A))
    k = 0
    for xq in Iterators.product(Xrs...)

        #nbs = getball(sz_A, node, graph, b_x)
        box_ranges = findsubgrid3(Xrs, b_x, xq)
        box_CIs = CartesianIndices(box_ranges)
        
        k += 1
        vs[k] = var( A[i] for i in box_CIs ; corrected = false )
    end

    return vs
end

function evalcost_hyperoptim(buffers, p, θ_ref, s_map, σ²::T) where T

    θ = updatekernel(θ_ref, p)
    score = zero(T)
    for i in eachindex(buffers)
        score += evalnlli!(
            buffers[i], θ, s_map, σ²,
        )
    end
    return sum(score)
end
