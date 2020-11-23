using Base.Tapir

function f()
    let token = @syncregion()
        @spawn token begin
            1 + 1
        end
        @sync_end token
    end
end

function taskloop(N)
    let token = @syncregion()
        for i in 1:N
            @spawn token begin
                1 + 1
            end
        end
        @sync_end token
    end
end

function taskloop2(N)
    @sync for i in 1:N
        @spawn begin
            1 + 1
        end
    end
end

function taskloop3(N)
    @par for i in 1:N
        1+1
    end
end

function vecadd!(out, A, B)
    @assert length(out) == length(A) == length(B)
    @inbounds begin
        @par for i in 1:length(out)
            out[i] = A[i] + B[i]
        end
    end
    return out
end

function fib(N)
    if N <= 1
        return N
    end
    token = @syncregion()
    x1 = Ref{Int64}()
    @spawn token begin
        x1[]  = fib(N-1)
    end
    x2 = fib(N-2)
    @sync_end token
    return x1[] + x2
end

###
# Interesting corner cases and broken IR
###

##
# Parallel regions with errors are tricky
# #1  detach within %sr, #2, #3
# #2  ...
#     unreachable()
#     reattach within %sr, #3
# #3  sync within %sr
#
# Normally a unreachable get's turned into a ReturnNode(),
# but that breaks the CFG. So we need to detect that we are
# in a parallel region.
#
# Question:
#   - Can we elimante a parallel region that throws?
#     Probably if the sync is dead as well. We could always
#     use the serial projection and serially execute the region.

function vecadd_err(out, A, B)
    @assert length(out) == length(A) == length(B)
    @inbounds begin
        @par for i in 1:length(out)
            out[i] = A[i] + B[i]
            error()
        end
    end
    return out
end

# This function is broken due to the PhiNode
@noinline function fib2(N)
    if N <= 1
        return N
    end
    token = @syncregion()
    x1 = 0
    @spawn token begin
        x1  = fib2(N-1)
    end
    x2 = fib2(N-2)
    @sync_end token
    return x1 + x2
end


module ReturnViaRef
using Base: Tapir

@noinline produce() = P
P = 1

@noinline function store!(d, x)
    d[] = x
    return
end

function f()
    a = Ref{Any}()
    local b
    Tapir.@sync begin
        Tapir.@spawn begin
            store!(a, produce())
        end
        b = produce()
    end
    return (a[], b)
end

function g()
    a = Ref{Any}()
    local b
    Tapir.@sync begin
        Tapir.@spawn begin
            a[] = produce()
        end
        b = produce()
    end
    return (a[], b)
end
end # module ReturnViaRef


module DecayedPointers
using Base: Tapir

mutable struct M2{T}
    x::Int
    y::T
end

@noinline function change!(m)
    m.y = 0
    return
end

function f()
    a = M2(1, M2(2, 3))
    b = M2(4, 5)
    Tapir.@sync begin
        Tapir.@spawn begin
            change!(a.y)
        end
        change!(b)
    end
    return (a, b)
end
end # module DecayedPointers


module SyncInLoop
using Base: Tapir

@noinline consume(x) = (global G = x; nothing)

function body0()
    x = (Ref(1), Ref(2))
    Tapir.@sync begin
        Tapir.@spawn consume(x)
        consume(x)
    end
end

function loop0(n)
    for _ in 1:n
        body0()
    end
end
end # module SyncInLoop


module NestedAggregates
using Base: Tapir

@noinline oneone() = [1]
@noinline twotwooneone() = ((oneone(), oneone()), (oneone(), oneone()))

function f()
    a = Ref{Any}()
    local b
    Tapir.@sync begin
        Tapir.@spawn begin
            a[] = twotwooneone()
        end
        b = twotwooneone()
    end
    return (a[], b)
end
end


module TaskOutputs
using Base: Tapir

@noinline produce() = P::Int
P = 1

function f()
    a = Tapir.Output()
    v = 'a'
    local b
    Tapir.@sync begin
        Tapir.@spawn begin
            a[] = (v, produce())
        end
        b = produce()
    end
    return (a[], b)
end
end # module TaskOutputs


function mapfold(f, op, xs; basesize = cld(length(xs), Threads.nthreads()))
    basesize = max(3, basesize)  # for length(left) + length(right) >= 4
    if length(xs) < basesize
        return mapfoldl(f, op, xs)
    end
    return _mapfold(f, op, xs, basesize)
end

function _mapfold(f, op, xs, basesize)
    if length(xs) <= basesize
        acc = @inbounds op(f(xs[begin]), f(xs[begin+1]))
        for i in eachindex(xs)[3:end]
            acc = op(acc, f(@inbounds xs[i]))
        end
        return acc
    else
        left = @inbounds @view xs[begin:(end-begin+1)÷2]
        right = @inbounds @view xs[(end-begin+1)÷2+1:end]
        ref = Tapir.Output()
        token = @syncregion()
        @spawn token begin
            ref[] = _mapfold(f, op, right, basesize)
        end
        y = _mapfold(f, op, left, basesize)
        @sync_end token
        return op(y, ref[])
    end
end

function append!!(a, b)
    ys::Vector = a isa Vector ? a : collect(a)
    if eltype(b) <: eltype(ys)
        zs = append!(ys, b)
    else
        zs = similar(ys, promote_type(eltype(ys), eltype(b)), (length(ys) + length(b)))
        copyto!(zs, 1, ys, 1, length(ys))
        zs[length(ys)+1:end] .= b
    end
    return zs
end

function tmap(f, xs; kw...)
    ys = mapfold(tuple ∘ f, append!!, xs; kw...)
    if ys isa Tuple
        return collect(ys)
    else
        return ys
    end
end
