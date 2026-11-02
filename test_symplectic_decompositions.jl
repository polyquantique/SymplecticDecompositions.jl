using Test
using StatsBase
include("SymplecticDecompositionsMod.jl")


@testset "Test issymplectic() returns true with example matrices" begin
    for ell in 1:3
        Omega = [zeros(ell, ell) I(ell); -I(ell) zeros(ell, ell)]
        @test issymplectic(Omega) == true
        @test issymplectic(I(2 * ell)) == true

        A = randn(ell, ell)
        S = [A zeros(ell, ell); zeros(ell, ell) inv(transpose(A))]
        @test issymplectic(S) == true

        U = rand(Haar(2), ell)
        O = [real(U) -imag(U); imag(U) real(U)]
        @test issymplectic(O) == true
    end
end

@testset "Test issymplectic() returns false for non-square matrices" begin
    for _ in 1:3
        pair = sample(1:100, 2, replace=false)
        A = randn(pair...)
        @test issymplectic(A) == false
    end
end

@testset "Test random_symplectic() matrix properties" begin
    for ell in 1:3
        for prec in 2:1:3
            S = random_symplectic(ell; prec=prec)
            @test issymplectic(S) == true
            @test isapprox(det(S), 1.0; atol=1e-10)
        end
    end
end

@testset "Test takagi() matrix properties" begin
    for ell in 2:4
        for prec in 2:1:3
            reA = Array{MultiFloat{Float64, prec}}(randn(ell, ell))
            imA = Array{MultiFloat{Float64, prec}}(randn(ell, ell))
            A = reA + im .* imA
            A = (A + transpose(A)) / 2

            D, W = takagi(A)

            @test isapprox(A, W * Diagonal(D) * transpose(W); atol=1e-10)
            @test isapprox(W' * W, I(ell); atol=1e-10)
            @test all(D .>= 0)
        end
    end
end

@testset "Test takagi() for real, symmetric, positive-definite matrices" begin
    for ell in 2:4
        for prec in 2:1:3
            A = Array{MultiFloat{Float64, prec}}(randn(ell, ell))
            A = (A + transpose(A)) / 2
            Ap = I(ell) + A * transpose(A)

            D, W = takagi(Ap)
            U, Dp, V = svd(Ap)

            @test isapprox(D, Dp; atol=1e-10)
            @test isapprox(W, U; atol=1e-10)
            @test isapprox(W, V; atol=1e-10)
        end
    end
end

@testset "Test bloch_messiah() matrix properties" begin
    for ell in 2:4
        for prec in 2:1:3
            S = random_symplectic(ell; prec=prec)

            O1, D, O2 = bloch_messiah(S)

            @test isapprox(S, O1 * Diagonal(D) * O2; atol=1e-10)
            @test issymplectic(O1) == true
            @test issymplectic(O2) == true
            @test issymplectic(Diagonal(D)) == true
            @test isapprox(O1 * transpose(O1), I(2 * ell); atol=1e-10)
            @test isapprox(O2 * transpose(O2), I(2 * ell); atol=1e-10)
        end
    end
end

@testset "Test bloch_messiah() on identity matrix" begin
    for ell in 2:4
        for prec in 2:1:3
            S = I(2 * ell)

            O1, D, O2 = bloch_messiah(S)

            @test isapprox(O1 * O2, I(2 * ell); atol=1e-10)
            @test isapprox(Diagonal(D), I(2 * ell); atol=1e-10)
            @test issymplectic(O1) == true
            @test issymplectic(O2) == true
            @test isapprox(O1 * transpose(O1), I(2 * ell); atol=1e-10)
            @test isapprox(O2 * transpose(O2), I(2 * ell); atol=1e-10)
        end
    end
end