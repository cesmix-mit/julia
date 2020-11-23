module TestTapir
include("tapir.jl")

@eval Base.Tapir _should_assert() = true

using Test

macro test_error(expr)
    @gensym err tmp
    quote
        local $err
        $Test.@test try
            $expr
            false
        catch $tmp
            $err = $tmp
            true
        end
        $err
    end |> esc
end

@testset "fib" begin
    @test fib(1) == 1
    @test fib(2) == 1
    @test fib(3) == 2
    @test fib(4) == 3
    @test fib(5) == 5
end

@testset "vecadd!" begin
    @test vecadd!([0], [1], [2]) == [3]
    @testset for n in [10, 1000, 3000]
        A = randn(n)
        B = randn(n)
        out = zero(A)
        @test vecadd!(out, A, B) == A .+ B
    end
end

@testset "return via Ref" begin
    @test ReturnViaRef.f() == (1, 1)
    @test ReturnViaRef.g() == (1, 1)
end

@testset "decayed pointers" begin
    @test begin
        a, b = DecayedPointers.f()
        (a.y.y, b.y)
    end == (0, 0)
end

@testset "sync in loop" begin
    @test (SyncInLoop.loop0(1); true)
    @test (SyncInLoop.loop0(3); true)
end

@testset "nested aggregates" begin
    x = NestedAggregates.twotwooneone()
    desired = (x, x)
    @test NestedAggregates.f() == desired
end

@testset "@spawn syntax" begin
    function setindex_in_spawn()
        ref = Ref{Any}()
        Tapir.@sync begin
            Tapir.@spawn ref[] = (1, 2)
        end
        return ref[]
    end
    @test setindex_in_spawn() == (1, 2)

    function let_in_spawn()
        a = 1
        b = 2
        ref = Ref{Any}()
        Tapir.@sync begin
            Tapir.@spawn let a = a, b = b
                ref[] = (a, b)
            end
        end
        return ref[]
    end
    @test let_in_spawn() == (1, 2)
end

@testset "`Tapir.Output`" begin
    @test @inferred(TaskOutputs.f()) == (('a', 1), 1)
    @test @inferred(tmap(x -> x + 0.5, 1:10)) == 1.5:1:10.5
end

@noinline always() = rand() <= 1
@noinline donothing() = always() || error("unreachable")

@testset "exceptions" begin
    function f()
        token = @syncregion()
        @spawn token begin
            always() && throw(KeyError(1))
        end
        @spawn token begin
            always() && throw(KeyError(2))
        end
        donothing()  # TODO: don't
        # `donothing` here is required for convincing the compiler to _not_
        # inline the tasks.
        @sync_end token
    end
    err = @test_error f()
    @test err isa CompositeException
    @test all(x -> x isa TaskFailedException, err.exceptions)
    exceptions = [e.task.result for e in err.exceptions]
    @test Set(exceptions) == Set([KeyError(1), KeyError(2)])
end

end
