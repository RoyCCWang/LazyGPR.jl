# Function References

## Hyperparameter optimization
```@docs
MarginalLikelihood
LOOCV
HCostConfig
LazyHCostConfig
MetaheuristicsConfig
KernelOptimConfig
UseMetaheuristics
OptimContainer
create_hp_cost
free_hp_cost
optimize_kernel_hp
optimize_kernel_hp_separately
```

## Warp sample construction
```@docs
create_grid_warp_samples
create_warp_samples
AxisGraph
KNNGraph
UseRieszDSP
UseSpatialGSP
UseInterpolations
UseScatteredInterpolation
```
## Kernels
```@docs
WendlandSplineKernel
Order1
Order2
Order3
SqExpKernel
DEKernel
evalkernel
evalde
evalwarpmap
```

## Model specification
```@docs
AdjustmentMap
GPData
create_local_procs
evalsmap
LazyGP
fitGP
fitGP!
```

## Model query
```@docs
QueryOptions
computecachevars
setup_query_dc
query_dc
free_query_dc
queryGP
```
