# Show the border issues for Riesz-wavelet and one-hop operators.
# Run `a.jl` first.

import RieszDSP as RZ

import Random
Random.seed!(25)

import Images
using LinearAlgebra
using Serialization

const T = Float64
const D = 2;

using BenchmarkTools
import VisualizationBag as VIZ
import PythonPlot as PLT
PLT.close("all")
fig_num = 1

# helper functions.
include("helpers/utils.jl")
include("helpers/image.jl")

# where the results are saved.
save_results_dir = joinpath(
    homedir(),
    "work/LazyGPR.jl/results/"
)
if !ispath(save_results_dir)
    mkpath(save_results_dir)
end

# user inputs
pics_folder = joinpath(homedir(), "work/data/images/kodak/")
image_path = joinpath(pics_folder, "kodim23.png")

img = loadkodakimage(T, image_path; discard_pixels = 1)
x_nD, x_ranges = image2samples(img)
sz_x = size(x_nD)

# name image data.
Xrs = (1:size(x_nD,1), 1:size(x_nD,2))
im_y = x_nD
x = vec(x_nD)

# # Graph laplacian

#graph, edge_list = GSP.getUnitGridgraph(nbs) # debug.
nbs = GSP.getgridnbs(size(x_nD))
G = GSP.UnitGrid(nbs)
graph = GSP.getgraph(G)

# Basic one-hop operators:
A = GSP.create_adjacency(G) # adjacency matrix
deg = GSP.create_degree(G) # degree matrix
deg_inv = GSP.create_invdegree(T, G)
L = GSP.create_laplacian(G) # combinatorial Laplacian matrix

# Normalized Laplacian operators.
Ln = GSP.create_snlaplacian(T, G) # symmetric normalized Laplacian matrix
TL = GSP.create_rwlaplacian(T, G) # random-walk Laplacian matrix

# Apply operators to signal.
Ax = A*x
Anx_pre = deg_inv*A*x
Anx_post = A*deg_inv*x
Lx = L*x
Tx = TL*x
Ln_x = Ln*x

# Riesz warp samples.
W_rz = LGP.create_grid_warp_samples(LGP.UseRieszDSP(RZ), im_y)

# # Visualize Border
# Specify border
function getcloseup(Xrs, Y, w, h)
    Y_out = Y[end-h:end, 220:220+w]
    Xrs_out = (Xrs[begin][end-h:end], Xrs[end][220:220+w])
    return Xrs_out, Y_out
end

width = 20 # measured in pixels
height = 10 # measured in pixels
Xrs_close, x_close = getcloseup(Xrs, x_nD, width, height)
_, Ax_close = getcloseup(Xrs, reshape(Ax, sz_x), width, height)
_, Anx_pre_close = getcloseup(Xrs, reshape(Anx_pre, sz_x), width, height)
_, Anx_post_close = getcloseup(Xrs, reshape(Anx_post, sz_x), width, height)
_, Lx_close = getcloseup(Xrs, reshape(Lx, sz_x), width, height)
_, Lnx_close = getcloseup(Xrs, reshape(Ln_x, sz_x), width, height)
_, Tx_close = getcloseup(Xrs, reshape(Tx, sz_x), width, height)
_, W_rz_close = getcloseup(Xrs, W_rz, width, height)

# plot
fig_size = VIZ.getaspectratio(size(im_y)) .* 4
dpi = 300
#dpi = 96

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    x_close,
    [],
    "x",
    fig_num,
    "Image data, x";
    cmap = "gray",
    vmin = 0,
    vmax = 1,
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
)

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Ax_close,
    [],
    "x",
    fig_num,
    "Ax";
    cmap = "gray",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    vmin = 0,
)

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Anx_pre_close,
    [],
    "x",
    fig_num,
    "D⁻¹Ax";
    cmap = "gray",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    vmin = 0,
    vmax = 1,
)

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Anx_post_close,
    [],
    "x",
    fig_num,
    "AD⁻¹x";
    cmap = "gray",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    vmin = 0,
    vmax = 1,
)

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Lx_close,
    [],
    "x",
    fig_num,
    "Lx";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    symmetric_color_range = true,
)

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Lnx_close,
    [],
    "x",
    fig_num,
    "ℒx";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    symmetric_color_range = true,
)

fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    Tx_close,
    [],
    "x",
    fig_num,
    "Tx";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    symmetric_color_range = true,
)


fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs_close),
    W_rz_close,
    [],
    "x",
    fig_num,
    "Warp samples via HWRT";
    cmap = "bwr",
    matrix_mode = true,
    fig_size = fig_size,
    dpi = dpi,
    symmetric_color_range = true,
)

nothing
