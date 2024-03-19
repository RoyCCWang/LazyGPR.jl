# for use with extensions

abstract type SolverConfig end

################

"""
    struct UseMetaheuristics
This is a trait data type that indicates Metaheuristics.jl should be used for optimization.
Usage:
```
import Metaheuristics as EVO
opt_trait = UseMetaheuristics(EVO)
```
"""
struct UseMetaheuristics <: ExtensionPkgs
    pkg::Module
end

"""
    @kwdef struct MetaheuristicsConfig <: SolverConfig
        f_calls_limit::Integer = 1_000
    end
The optimization configuration for Metaheuristics.jl's 
"""
@kwdef struct MetaheuristicsConfig <: SolverConfig
    f_calls_limit::Integer = 1_000
end

function loaderr_ext_opt_pkg()
    error("To use optimize_kernel_hp(), please load one of the following optimization packages: Metaheuristics.jl")
end

# single-process version. Used for conventional GP.
function run_optimizer(
    alg_trait::UseMetaheuristics,
    args...
    )

    alg = alg_trait.pkg
    ext_evo = Base.get_extension(@__MODULE__, :MetaheuristicsExt)

    # avoid type instability; write one for each extension.
    proceed_flag = !isnothing(ext_evo) && alg == ext_evo.Metaheuristics
    if !proceed_flag
        loaderr_ext_opt_pkg()
    end
    
    return _run_optimizer(alg_trait, args...)
end

function _run_optimizer(::Nothing, args...)
    return loaderr_ext_opt_pkg()
end

####