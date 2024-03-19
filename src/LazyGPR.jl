module LazyGPR

using Pkg
using Distributed
using LinearAlgebra
using Serialization
using Statistics

import Distances
import NearestNeighbors as NN

import Roots
import DistributedData as DData

include("types.jl")

include("distributed/batch_grid.jl")
include("frontend/query.jl")

include("kernels/basic.jl")
include("kernels/dek.jl")
include("kernels/matrix.jl")

include("warpmap.jl")
include("re_grid.jl")

include("models/conventional.jl")

include("distributed/IO.jl")

include("models/obs_noise.jl")
include("models/lazy/buffers.jl")
include("models/lazy/hoptim.jl")
include("models/lazy/q2.jl")

include("frontend/solvers.jl")
include("frontend/model_selection.jl")

include("utils.jl")

export create_grid_warp_samples, # warp samples and function
create_warp_samples,
AxisGraph,
KNNGraph,
UseRieszDSP,
UseSpatialGSP,
UseInterpolations,
UseScatteredInterpolation,

# hyperparameter optimization
MarginalLikelihood,
LOOCV,
HCostConfig,
LazyHCostConfig,
MetaheuristicsConfig,
KernelOptimConfig,
UseMetaheuristics,
OptimContainer,
create_hp_cost,
free_hp_cost,
optimize_kernel_hp,
optimize_kernel_hp_separately,

# kernels
WendlandSplineKernel,
Order1,
Order2,
Order3,
SqExpKernel,
DEKernel,
evalkernel,
evalde,
evalwarpmap,
fitGP!,

# GPR model specification
GPData,
create_local_procs,
evalsmap,
AdjustmentMap,
LazyGP,
fitGP,

# query
QueryOptions,
computecachevars,
setup_query_dc,
query_dc,
free_query_dc,
queryGP

end # module LazyGPR
