module MetaheuristicsExt

using LazyGPR
const LGP = LazyGPR

import Metaheuristics
const EVO = Metaheuristics

include("helpers/evo.jl")


function LGP._run_optimizer(
    ::LGP.UseMetaheuristics,
    costfunc,
    lbs::Vector{T},
    ubs::Vector{T},
    solver_config::LGP.MetaheuristicsConfig,
    p0s::Vector{Vector{T}},
    ) where T <: AbstractFloat

    f_calls_limit = solver_config.f_calls_limit

    evo_config = ECAConfig(f_calls_limit, p0s) #p0)
    result = runevofull(costfunc, lbs, ubs, evo_config)
    sol_vars = EVO.minimizer(result)
    sol_cost = EVO.minimum(result)

    return sol_vars, sol_cost
end

end