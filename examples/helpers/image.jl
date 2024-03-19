
function loadkodakimage(::Type{T}, file_path::String; discard_pixels::Integer = 0) where T <: AbstractFloat
    
    img = Images.load(file_path)
    gray_img = Images.Gray.(img)
    y_nD = convert(Array{T}, gray_img)
    
    #some kodak images have border artefacts. remove pixels.
    a = discard_pixels
    y_nD = y_nD[begin+a:end-a, begin+a:end-a]    

    return y_nD
end

function image2samples(y_nD::Matrix{T}) where T

    Nr, Nc = size(y_nD)
    x_ranges = getmatrixranges(Nr, Nc)
    y = y_nD

    return y, x_ranges
end

function getmatrixranges(Nr::Integer, Nc::Integer)

    v_range = LinRange(1, Nr, Nr)
    h_range = LinRange(1, Nc, Nc)
    
    x_ranges = Vector{LinRange{T,Int}}(undef, 2)
    x_ranges[1] = v_range
    x_ranges[2] = h_range

    return x_ranges
end

function getdownsampledimage(
    ::Type{T},
    image_path,
    down_factor::Integer;
    discard_pixels = 1) where T
    
    img = loadkodakimage(T, image_path; discard_pixels = discard_pixels)
    im_y_ref = convert(Array{T}, Images.Gray.(img))
    
    
    im_y = im_y_ref[begin:down_factor:end, begin:down_factor:end]
    image_ranges = getmatrixranges(size(im_y)...)

    return im_y, image_ranges, im_y_ref
end

function bird_bounds(down_factor)

    query_b_box = LGP.BoundingBox(
        round.(Int, (200, 190) ./ down_factor),
        round.(Int, (275, 261) ./ down_factor),
    )
    file_name = "kodim23.png"

    return file_name, query_b_box
end

function helmet_bounds(down_factor)

    query_b_box = LGP.BoundingBox(
        #round.(Int, (42, 36) ./ down_factor),
        #round.(Int, (192, 166) ./ down_factor),
        round.(Int, (36, 42) ./ down_factor), # reverse the coordinates from GIMP.
        round.(Int, (166, 192) ./ down_factor),
    )
    file_name = "kodim05.png"

    return file_name, query_b_box
end

function propeller_bounds(down_factor)

    query_b_box = LGP.BoundingBox(
        #round.(Int, (138, 60) ./ down_factor),
        #round.(Int, (208, 210) ./ down_factor),
        round.(Int, (60, 138) ./ down_factor),
        round.(Int, (210, 208) ./ down_factor),
    )
    file_name = "kodim20.png"

    return file_name, query_b_box
end

function load_kodak_region(
    tag::String,
    data_dir::String,
    b_x::T;
    down_factor::Integer = 2,
    up_factor::Real = 4.5,
    ) where T <: AbstractFloat
    
    file_name, query_b_box = bird_bounds(down_factor)
    if tag == "propeller"
        
        file_name, query_b_box = propeller_bounds(down_factor)
    
    elseif tag == "helmet"
        
        file_name, query_b_box = helmet_bounds(down_factor)
    end

    im_y, image_ranges, _ = getdownsampledimage(
        T, joinpath(data_dir, file_name), down_factor;
        discard_pixels = 0,
    )

    xq_ranges = LGP.getqueryranges(
        up_factor, query_b_box, step.(image_ranges),
    )
    
    #noise.
    M = floor(Int, b_x) #base this on b_x.
    L = M #must be an even positive integer. The larger the flatter.
    if isodd(L)
        L = L + 1
    end
    x0, y0 = convert(T, 0.8*M), 1 + convert(T, 0.5)
    s_map = LGP.AdjustmentMap(x0, y0, b_x, L)

    return im_y, Tuple(image_ranges), s_map, Tuple(xq_ranges)
end
