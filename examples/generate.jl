using Literate

include("helpers/gen_utils.jl")

# `dest_dir` is where the generated files will end up in. We delete all the files in that directory first.
dest_dir = "../docs/src/generated"
reset_dir(dest_dir) # creates the path if it doesn't exist.

# # Bernstein filtering
# fix the URL. This is generated because we're using Documenter.jl-flavoured Markdown.
postprocfunc = xx->replace(
    xx,
    "EditURL = \"compare_warpmaps.jl\"" =>
    "EditURL = \"../../../examples/compare_warpmaps.jl\"" # as ifthe pwd() is in the `dest_dir`
)

Literate.markdown(
    "compare_warpmaps.jl";
    execute = true,
    name = "compare_warpmaps_lit", # make this different ahn "bernstein_filter.jl" so it is easier to find and delete all generated files.
    postprocess = postprocfunc,
)

move_prefix_name = "compare_warpmaps_lit"
move_md(dest_dir, move_prefix_name)


# # Adjustment Map
postprocfunc = xx->replace(
    xx,
    "EditURL = \"s_map.jl\"" =>
    "EditURL = \"../../../examples/s_map.jl\"" # as ifthe pwd() is in the `dest_dir`
)

Literate.markdown(
    "s_map.jl";
    execute = true,
    name = "s_map_lit", # make this different ahn "bernstein_filter.jl" so it is easier to find and delete all generated files.
    postprocess = postprocfunc,
)

move_prefix_name = "s_map_lit"
move_md(dest_dir, move_prefix_name)

# # Rainfall
postprocfunc = xx->replace(
    xx,
    "EditURL = \"rainfall.jl\"" =>
    "EditURL = \"../../../examples/rainfall.jl\"" # as ifthe pwd() is in the `dest_dir`
)

Literate.markdown(
    "rainfall.jl";
    execute = true,
    name = "rainfall_lit", # make this different ahn "bernstein_filter.jl" so it is easier to find and delete all generated files.
    postprocess = postprocfunc,
)

move_prefix_name = "rainfall_lit"
move_md(dest_dir, move_prefix_name)

# We didn't include upconvert.jl because it uses distributed computing, which Literate.jl has some issues with thread locking.

nothing

