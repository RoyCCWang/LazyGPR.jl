using Documenter
using LazyGPR

# # local.
makedocs(
    sitename = "LazyGPR",
    modules = [LazyGPR],
    #format = Documenter.HTML(),
    pages = [
        "Overview" => "index.md",
        "Public API" => "api.md",
        
        "Visualize Warp Samples" =>
        "generated/compare_warpmaps_lit.md",

        "Visualize Adjustment Map" =>
        "generated/s_map_lit.md",

        "Demo: Oscillation Reduction" =>
        "generated/rainfall_lit.md",
    ],
)

# github.
makedocs(
    sitename="LazyGPR.jl",
    modules=[LazyGPR],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing)=="true"),
    pages = [
        "Overview" => "index.md",
        "Public API" => "api.md",
        
        "Visualize Warp Samples" =>
        "generated/compare_warpmaps_lit.md",

        "Visualize Adjustment Map" =>
        "generated/s_map_lit.md",

        "Demo: Oscillation Reduction" =>
        "generated/rainfall_lit.md",
    ],
)
deploydocs(
    repo = "github.com/RoyCCWang/LazyGPR.jl",
    target = "build",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#" ],
)