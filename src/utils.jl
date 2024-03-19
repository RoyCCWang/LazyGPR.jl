
function getunitranges(::Type{T}, sz::NTuple{D, Int}) where {T, D}
    
    x_ranges = collect(
        (one(T)):(convert(T, sz[d]))
        for d in eachindex(sz)
    )
    return Tuple(x_ranges)
end

function resample_intervals(Xrs::NTuple{D,T}, Ns::Tuple) where {D, T}
    ntuple(d->LinRange(first(Xrs[d]), last(Xrs[d]), Ns[d]), D)
end

# toy example.
function showsize(dataset)
    return map(
        fetch,
        [
            DData.get_from(
                w,
                :(
                    (size(Bs_wk), length(constants_wk), size(p_wk))
                )
            )
            for w in dataset.workers
        ]
    )
end

"""
    create_local_procs(
        n_new_workers::Integer;
        pkg_load_list::Vector{Expr} = Vector{Expr}(undef, 0),
        verbose = true,
    )

Creates `n_new_workers` many new workers, and loads the packages in `pkg_load_list` on each worker.

Please manually check whether your current Julia session can actually benefit from launching `n_new_workers` new workers. For example, the number of workers should not exceed the number of physical floating-point cores in your distributed computing environment.
"""
function create_local_procs(
    n_new_workers::Integer;
    pkg_load_list::Vector{Expr} = Vector{Expr}(undef, 0),
    verbose = true,
    )

    # TODO We could warn the user if the number (current of to-be-added) number of workers exceed Sys.CPU_THREADS
    # async create n_workers worker processes

    N_workers = Distributed.nworkers()
    st = N_workers+1
    fin = st + n_new_workers - 1

    new_pids = Vector{Int}(undef, 0)
    tasks = map(st:fin) do n
        @async create_local_proc(
            new_pids;
            pkg_load_list = pkg_load_list,
            n = n,
            verbose = verbose,
        )
    end

    t = @async map(fetch, tasks)
    wait(t)

    return new_pids
end

function create_local_proc(
    new_pids::Vector{Int};
    pkg_load_list::Vector{Expr} = Vector{Expr}(undef, 0),
    n = nothing,
    verbose = true,
    )

    n_str = n === nothing ? "" : " $(n)"
    worker_str = string("worker", n_str)

    if verbose
        @info "Requesting $(worker_str)..."
    end
    pid = only(addprocs(1))
    #@show typeof(pid), pid
    push!(new_pids, pid)
    
    # make sure we activate the ACTUAL PROJECT that's active on the manager,
    # which may be different than `@.` during e.g. CI runs
    project = Pkg.project().path
    Distributed.remotecall_eval(
        Main,
        pid,
        :(using Pkg; Pkg.activate($(project))),
    )

    if !isempty(pkg_load_list)
        # convert expressions to strings.
        titles = String.(Symbol.(pkg_load_list))
        
        for i in eachindex(pkg_load_list)
            label = titles[i]
            ex = pkg_load_list[i]

            if verbose
                @info "Loading $label on $(worker_str)..."
            end
            Distributed.remotecall_eval(
                Main,
                pid,
                ex,
            )
        end
    end

    if verbose
        @info "Loading LazyGPR.jl on $(worker_str)..."
    end
    Distributed.remotecall_eval(
        Main,
        pid,
        :(import LazyGPR),
    )

    if verbose
        @info "$(worker_str) ready, PID $(pid)"
    end
    return pid
end

function free_local_procs(worker_list::Union{Vector,NTuple})

    t = rmprocs(worker_list..., waitfor=0)
    wait(t)
    return nothing
end

function scaleranges(xrs::NTuple{D, AR}, m::Integer) where {D, AR <: AbstractRange}

    return ntuple(
        d->LinRange(first(xrs[d]), last(xrs[d]), m*length(xrs[d])),
        D,
    )
end

function ranges2matrix(Xrs::NTuple{D, AR}) where {D, AR <: AbstractRange}
    it_prod = Iterators.product(Xrs...)
    v = collect( Iterators.flatten(it_prod) )
    return reshape(v, D, length(it_prod))
end

function getboxbounds(X::Vector{AV}) where AV <: AbstractVector
    
    @assert !isempty(X)
    D = length(X[begin])
    
    lbs = collect(
        minimum( X[n][d] for n in eachindex(X) )
        for d in 1:D
    )
   
    ubs = collect(
        maximum( X[n][d] for n in eachindex(X) )
        for d in 1:D
    )
    
    return lbs, ubs
end

function vecs2mat(X::Vector)
    @assert !isempty(X)

    D = length(X[begin])
    return reshape(collect(Iterators.flatten(X)), D, length(X))
end