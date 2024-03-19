using Test
using LinearAlgebra

import Random
Random.seed!(25)

import LazyGPR as LGP

include("../examples/helpers/synth_data.jl") # setupAmatrix()
include("../examples/helpers/utils.jl") # setupAmatrix()

@testset "bandwidth adjustment: half width at half max" begin

    T = Float64
    zero_tol = convert(T, 1e-7)
    N_tests = 200
    scale_factor = 50 # for h.

    # set up constants.
    a_lb = convert(T, 1e-3)
    a_ub = convert(T, 10)

    # the kernels that we test.
    sqexp = LGP.SqExpKernel(one(T))
    spline32 = LGP.WendlandSplineKernel(
        LGP.Order2(), one(T), 3,
    )
    kernels = [sqexp; spline32]
    
    # tests.
    for sk in kernels
        for _ = 1:N_tests
            
            # generate parameters for this test.
            h = abs(scale_factor*rand())

            h_sk = rand()
            if isapprox(h_sk, one(T)) || isapprox(h_sk, zero(T))
                h_sk = rand()
            end

            # run the function under test.
            a_star, x_star, status = LGP.findbandwidth(
                sk, h, a_lb, a_ub;
                h_sk = h_sk,
            )

            # check.
            if status
                sk_x = LGP.evalkernel(x_star, sk)
                @test isapprox(sk_x, h_sk)

                sk2 = LGP.createkernel(sk, a_star)
                sk2_x = LGP.evalkernel(x_star*h, sk2)
                @test isapprox(sk2_x, h_sk)
            end
        end
    end

end