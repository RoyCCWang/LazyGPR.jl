# In this demo, we go over an image-upconversion example. We fit the hyperparameters for a lazy-evaluation GPR, querying it to up-convert an image.

# # Setup
# Install dependencies
import Pkg
let
    pkgs = ["PythonPlot", "VisualizationBag", "Images", "SpatialGSP", "LocalFilters", "Interpolations", "Metaheuristics", "LazyGPR"]
    for pkg in pkgs
        if Base.find_package(pkg) === nothing
            Pkg.add(pkg)
        end
    end
end;

import Random
Random.seed!(25)

using LinearAlgebra

import Interpolations
import Metaheuristics as EVO
import Images
import LocalFilters
import SpatialGSP as GSP
import LazyGPR as LGP

import VisualizationBag as VIZ

import PythonPlot as PLT
PLT.close("all")
fig_num = 1

const T = Float64
const D = 2;

# # User Inputs
save_dir = "output/"
if !ispath(save_dir)
    mkpath(save_dir)
end

#increase to use distributed computing.
N_workers = 7
@assert N_workers > 1 #this script was designed for multi-process distributed computing.

#Data preparation
data_dir = "data/"
image_file_name = "kodim05_cropped.png"
model_selection_string = "ML" # "ML" means we use marginal likelihood for the hyperparameter optimization objective. Another valid option is "LOOCV", for the leave-one-out cross-validation objective; see  https://doi.org/10.7551/mitpress/3206.001.0001  for details on LOOCV.
down_factor = 2; # downsample/up-conversion factor.

#Prameters for smoothing the warp samples before making it into a warp function. The bilateral filter is used.
#If this is larger, then the warp samples are smoother, with increased risk of removing details:
#see compute_kodak_hp() and compute_kodak_warpmap() from the helper script `hopt.jl` to see how `σr_factor` is used to compute `σr`, the bilateral filter intensity standard deviation.
σr_factor = convert(T, 4.0)
σs = one(T); # The spatial standard deviation for the bilateral filter.

# Lazy-evaluation GPR model parameters:
σ² = convert(T, 0.001)
b_x = convert(T, 6);

#For hyperparameter optimization
f_calls_limit = 1_000 # soft-constraint on the number of objective function evaluations during optimization.
N_neighbourhoods = 50 # The maximum number of local datasets we use in the objective function is this number times two. This is `M` in our manuscript.
a_lb = convert(T, 0.001) # lower bound for the bandwidth parameter.
a_ub = one(T)
N_initials_a = 100 # Number of initial guesses for the a, bandwidth parameter.
N_initials_κ = 100; # Similarly for the κ, gain parameter.

#how the warp samples are combined across different graph spectrums.
aggregate_symbol = :sum # An alternative option is :sum_normalized.

if model_selection_string != "ML" &&
        model_selection_string != "LOOCV"

    println("Unknown model_selection_string. Default to Marginal likelihood.")
    model_selection_string = "ML"
end

model_selection_trait = LGP.MarginalLikelihood()
if model_selection_string == "LOOCV"
    model_selection_trait = LGP.LOOCV()
end

# ## Setup distributed environments
using Distributed

worker_list = ones(Int, 1) #if this is the first time we're launching.
if nprocs() > 1
    worker_list = workers() #if we're re-using the existing workers.
end
if nworkers() < N_workers
    worker_list = LGP.create_local_procs(
        N_workers;
        pkg_load_list = [
            :(import Interpolations);
            :(import Metaheuristics);
        ],
        verbose = false,
    )
end


# The following helper scripts can be found in `examples/helpers/` from the root repository folder.
include("helpers/hopt.jl")
include("helpers/utils.jl")
include("helpers/image.jl");

# Setup data.
data_path = joinpath(data_dir, image_file_name)
im_y, image_ranges, _ = getdownsampledimage(
    T, data_path, down_factor;
    discard_pixels = 0,
)
Xrs = Tuple(image_ranges);

# adjustment map.
M = floor(Int, b_x) # base this on b_x.
L = M # must be an even positive integer. The larger the flatter.
if isodd(L)
    L = L + 1
end
x0, y0 = convert(T, 0.8 * M), 1 + convert(T, 0.5)
s_map = LGP.AdjustmentMap(x0, y0, b_x, L);

# Warp samples, from the unit grid graph.
warp_config = GSP.WarpConfig{T}(
    aggregate_option = aggregate_symbol,
)
W = LGP.create_grid_warp_samples(
    LGP.UseSpatialGSP(GSP),
    im_y,
    warp_config,
);

# ## Unit grid vs. axis-search graph
# We commented the following block out because it takes about a minute to run on our test machine. Uncomment it to see that an axis-search graph would generate the same warp samples as the unit graph. The axis-search graph is O(DN^2), so we don't use it when we know the data is on a uniform grid.
#X = vec(
#    collect(
#        convert(Vector{T}, collect(x))
#        for x in Iterators.product(Xrs...)
#    )
#)
#y = vec(im_y)
#axis_config = GSP.AxisSearchConfig{T}(w_lb = convert(T, 0.1)) #the w_lb can be any finite positive number for this equivalence example.
#W_axis = LGP.create_warp_samples(
#    LGP.UseSpatialGSP(GSP),
#    LGP.AxisGraph(),
#    X, y,
#    axis_config,
#    warp_config,
#);
#Should get practical zero (up to finite-precision arithmetic of floating-point numbers). This is the discrepancy between the samples generated by the axis-search graph and the grid graph.
#norm(vec(W)-W_axis)
# The axis-graph construction actually generates a grid graph with weights set to whatever value `axis_config.w_lb` is. The warp samples use the random-walk Laplacian as its fundamental one-hop graph filter operator, which is a type of normalized Laplacian, so this graph with edge weights scaled by `w_lb` yields the same result as the result from using a unit grid graph.

# # Kerenl hyperparameter fitting

# Assemble model container.
model = LGP.LazyGP(
    b_x, s_map,
    LGP.GPData(σ², Xrs, im_y),
);

# kernel selection: we use a 3D, order 2 Wendland spline kernel.
ref_can_kernel = LGP.WendlandSplineKernel(
    LGP.Order2(), one(T), 3,
);

# Optimization tuning parameters.
max_abs_W = maximum(abs.(W))
κ_ub = convert(T, (b_x * 3) * 1 / max_abs_W)
V = im_y; # The variances for each local dataset.

#lazy-evaluation hyperparameter optimization config
lazy_hopt_config = LGP.LazyHCostConfig(
    worker_list, N_neighbourhoods, true, V,
);

#optimization algorithm config:
solver_config = LGP.MetaheuristicsConfig(
    f_calls_limit = f_calls_limit,
);

#optimization problem (e.g. bounds) config:
optim_config = LGP.KernelOptimConfig{T}( # optim_config
    a_lb = a_lb,
    a_ub = a_ub,
    κ_ub = κ_ub,
    width_factor = 1, # carry over sk.
);

# Fit the hyperparameters.
dek_vars, dek_score, sk_vars, sk_score = compute_kodak_hp(
    model, W, σr_factor, Xrs, ref_can_kernel,
    model_selection_trait,
    LGP.UseMetaheuristics(EVO),
    lazy_hopt_config,
    solver_config, optim_config,
);

# The gain κ and canonical kernel bandwidth a, for the DE kernel:
κ, a_DE = dek_vars

# The DE kernel's optimization solution's objective score:
dek_score

# The stationary kernel's bandwidth (should be the same as the canonical kernel if width_factor = 1 for KernelOptimConfig())
sk_vars

# The stationary kernel's optimization solution's objective score:
sk_score

# # Query GPR Models
# Setup options
options = LGP.QueryOptions();

up_factor = down_factor
Xqrs = (
    LinRange(first(Xrs[1]), last(Xrs[1]), round(Int, length(Xrs[1]) * up_factor)),
    LinRange(first(Xrs[2]), last(Xrs[2]), round(Int, length(Xrs[2]) * up_factor)),
)
Nr, Nc = length.(Xqrs);

# Canonical Kernel Model
mqs_sk_vec, vqs_sk_vec = upconvert_kodak_sk(
    sk_vars, worker_list, model, options, Xqrs,
)
mqs_sk = reshape(mqs_sk_vec, Nr, Nc) #predictive means.
vqs_sk = reshape(vqs_sk_vec, Nr, Nc); #predictive variances.

# DE Kernel Model
mqs_dek_vec, vqs_dek_vec = upconvert_kodak_dek(
    W, σr_factor, Xrs,
    dek_vars,
    worker_list, model, options,
    Xqrs,
)

mqs_dek = reshape(mqs_dek_vec, Nr, Nc) #predictive means.
vqs_dek = reshape(vqs_dek_vec, Nr, Nc); #predictive variances.

# # Query bi-cubic interpolator
#Setup the bi-cubic interpolator.
itp = Interpolations.interpolate(
    im_y,
    Interpolations.BSpline(
        Interpolations.Cubic(
            Interpolations.Line(Interpolations.OnGrid()),
        ),
    ),
)
scaled_itp = Interpolations.scale(
    itp, Xrs...,
)
etp = Interpolations.extrapolate(
    scaled_itp, zero(T),
);

#Evaluate at the query positions.
itp_Xq = collect(
    etp(x...)
        for x in Iterators.product(Xqrs...)
);

# # VBisualize Results
# Get the non-downsampled image.
oracle_image, _, _ = getdownsampledimage(
    T, data_path, 1;
    discard_pixels = 0,
);

fig_size = VIZ.getaspectratio(size(oracle_image)) .* 8
dpi = 300;

# Oracle image:
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    oracle_image,
    [],
    "x",
    fig_num,
    "Oracle image";
    cmap = "gray",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
    fig_size = fig_size,
)
PLT.gcf()

# DE Kernel, predictive mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_dek,
    [],
    "x",
    fig_num,
    "DE kernel - predictive mean";
    cmap = "gray",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
    fig_size = fig_size,
)
PLT.gcf()

# Bi-cubic interpolation.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    itp_Xq,
    [],
    "x",
    fig_num,
    "Interpolation";
    cmap = "gray",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
    fig_size = fig_size,
)
PLT.gcf()

# Canonical Kernel, predictive mean.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    mqs_sk,
    [],
    "x",
    fig_num,
    "Canonical kernel - predictive mean";
    cmap = "gray",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    display_color_bar = false,
    fig_size = fig_size,
)
PLT.gcf()

# DE Kernel, predictive variance.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    log.(vqs_dek),
    [],
    "x",
    fig_num,
    "DE kernel - preditive log-variance";
    cmap = "plasma",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.6,
    fig_size = fig_size,
)
PLT.gcf()

# Canonical Kernel, predictive variance.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xqrs),
    vqs_sk,
    [],
    "x",
    fig_num,
    "Canonical kernel - predictive variance";
    cmap = "plasma",
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.6,
    fig_size = fig_size,
)
PLT.gcf()

nothing
