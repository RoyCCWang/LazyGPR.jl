# conventional GPR vs. Lazy GPR.

PLT.close("all")
fig_num = 1

const T = Float64
const D = 2

image_path = joinpath("data", "camera_man.png")

function getmatrixranges(Nr::Integer, Nc::Integer)

    v_range = LinRange(1, Nr, Nr)
    h_range = LinRange(1, Nc, Nc)

    x_ranges = Vector{LinRange{T, Int}}(undef, 2)
    x_ranges[1] = v_range
    x_ranges[2] = h_range

    return x_ranges
end

img = Images.load(image_path)
gray_img = Images.Gray.(img)
y_nD = convert(Array{T}, gray_img)
Nr, Nc = size(y_nD)
x_ranges = getmatrixranges(Nr, Nc)
sz_y = size(y_nD);

# # Visualize.
Xrs = (1:size(y_nD, 1), 1:size(y_nD, 2))
im_y = y_nD
fig_size = VIZ.getaspectratio(size(im_y)) .* 4
dpi = 300;

# Input image.
fig_num = VIZ.plotmeshgrid2D(
    PLT,
    collect(Xrs),
    im_y,
    [],
    "x",
    fig_num,
    "Image data";
    cmap = "gray",
    vmin = 0,
    vmax = 1,
    horizontal_title = "",
    vertical_title = "",
    matrix_mode = true,
    color_bar_shrink = 0.7,
    fig_size = fig_size = fig_size,
    dpi = dpi,
)
PLT.gcf()


# if kodak didn't optim σ² , then don't optim it.


nothing
