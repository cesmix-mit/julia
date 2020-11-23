module Tapir

export @syncregion, @spawn, @sync_end, @par

using Base: _wait

#####
##### Julia-Tapir Runtime
#####

# const TaskGroup = Vector{Task}
const TaskGroup = Channel{Task}

struct OutlinedFunction
    f::Ptr{Cvoid}
    arg::Ptr{UInt8}
    buffer::Vector{UInt8}
    root::Vector{Any}
    world_age::Csize_t
end

(outlined::OutlinedFunction)() = Base.invoke_in_world(outlined.world_age, call, outlined)

function call(outlined::OutlinedFunction)
    ccall(outlined.f, Cvoid, (Ptr{UInt8},), outlined.arg)
    return nothing
end

@assert precompile(Tuple{OutlinedFunction})
@assert precompile(call, (OutlinedFunction,))

# taskgroup() = Task[]
taskgroup() = Channel{Task}(Inf)

function _should_assert()
    false
    # true  # uncomment to debug
end

@noinline function all_valid_julia_object(root::Vector{Any})
    taskid = repr(objectid(current_task()))
    for i in eachindex(root)
        p = unsafe_load(Ptr{Ptr{Nothing}}(pointer(root, i)))
        @debug "task=$taskid: checking `&root[$i] = $p`"
        if isassigned(root, i)
            v = root[i]
            t = typeof(v)
            trepr = repr(t)
            vsummary = summary(v)
            @debug """
            task=$taskid: `typeof(root[$i]) = $trepr`
            `summary(root[$i]) = $vsummary`
            """
        end
    end
    return true
end

const OUTLINE_BUFFER_ALIGNMENT = 32

function spawn!(tasks::TaskGroup, f::Ptr{Cvoid}, parg::Ptr{UInt8}, arg_size::UInt,
                root::Vector{Any}, world_age::Csize_t)
    _should_assert() && @assert all_valid_julia_object(root)
    buffer = Vector{UInt8}(undef, arg_size + OUTLINE_BUFFER_ALIGNMENT)
    overshoot = mod(UInt(pointer(buffer)), OUTLINE_BUFFER_ALIGNMENT)
    if overshoot == 0
        offset = 0
    else
        offset = OUTLINE_BUFFER_ALIGNMENT - overshoot
    end
    copyto!(buffer, 1 + offset, unsafe_wrap(Vector{UInt8}, parg, arg_size), 1, arg_size)
    t = Task(OutlinedFunction(f, pointer(buffer) + offset, buffer, root, world_age))
    t.sticky = false
    schedule(t)
    push!(tasks, t)
    return nothing
end

function sync!(tasks::TaskGroup)
    # We can use `while isempty(tasks)` without data race because once we hit
    # `isempty(tasks)`, there is no task adding a new task to this task group.
    c_ex = nothing
    while !isempty(tasks)
        r = popfirst!(tasks)
        _wait(r)
        if istaskfailed(r)
            if c_ex === nothing
                c_ex = CompositeException()
            end
            push!(c_ex, TaskFailedException(r))
        end
    end
    close(tasks)
    if c_ex !== nothing
        throw(c_ex)
    end
end

@assert precompile(taskgroup, ())
@assert precompile(spawn!, (TaskGroup, Ptr{Cvoid}, Ptr{UInt8}, UInt, Vector{Any}, Csize_t))
@assert precompile(sync!, (TaskGroup,))

#####
##### Julia-Tapir Frontend
#####

mutable struct Output
    x::Any
    Output() = new()
end

Base.getindex(o::Output) = o.x
Base.setindex!(o::Output, x) = o.x = x

macro syncregion()
    Expr(:syncregion)
end

macro spawn(token, expr)
    Expr(:spawn, esc(token), esc(Expr(:block, expr)))
end

macro sync_end(token)
    Expr(:sync, esc(token))
end

macro loopinfo(args...)
    Expr(:loopinfo, args...)
end

const tokenname = gensym(:token)
macro sync(block)
    var = esc(tokenname)
    quote
        let $var = @syncregion()
            $(esc(block))
            @sync_end($var)
        end
    end
end

macro spawn(expr)
    var = esc(tokenname)
    quote
        @spawn $var $(esc(expr))
    end
end

macro par(expr)
    par_impl(:dac, expr)
end

macro par(strategy::Symbol, expr)
    par_impl(strategy, expr)
end

function par_impl(strategy::Symbol, expr)
    @assert expr.head === :for
    stcode = get((dac = ST_DAC, seq = ST_SEQ), strategy, nothing)
    if stcode === nothing
        error("Invalid strategy: ", strategy)
    end
    token = gensym(:token)
    body = expr.args[2]
    lhs = expr.args[1].args[1]
    range = expr.args[1].args[2]
    quote
        let $token = @syncregion()
            for $(esc(lhs)) = $(esc(range))
                @spawn $token $(esc(body))
                $(Expr(:loopinfo, (Symbol("tapir.loop.spawn.strategy"), Int(stcode))))
            end
            @sync_end $token
        end
    end
end

"""
    Base.Tapir.SpawningStrategy

[INTERNAL] Tapir spawning strategies.

This type enumerates valid arguments to `tapir.loop.spawn.strategy` loopinfo.
For the C++ coutner part, see `TapirLoopHints::SpawningStrategy`.

See:
* https://github.com/OpenCilk/opencilk-project/blob/opencilk/beta3/llvm/include/llvm/Transforms/Utils/TapirUtils.h#L216-L222
* https://github.com/OpenCilk/opencilk-project/blob/opencilk/beta3/llvm/include/llvm/Transforms/Utils/TapirUtils.h#L256-L265
"""
@enum SpawningStrategy begin
    ST_SEQ  # Spawn iterations sequentially
    ST_DAC  # Use divide-and-conquer
end

end

# Runtime functions called via `../src/task-tapir.c`
const _Tapir_taskgroup = Tapir.taskgroup
const _Tapir_spawn! = Tapir.spawn!
const _Tapir_sync! = Tapir.sync!
