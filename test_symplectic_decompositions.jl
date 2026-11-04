using Test
using StatsBase
include("SymplecticDecompositions.jl")

@testset "Test issymplectic() throws for complex matrices" begin
    A = [1 + im 0; 0 1]
    @test_throws ArgumentError issymplectic(A)
end

@testset "Test issymplectic() returns true with example matrices" begin
    for ell = 1:3
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
    for _ = 1:3
        pair = sample(1:100, 2, replace = false)
        A = randn(pair...)
        @test issymplectic(A) == false
    end
end

@testset "Test premultiply_by_Omega() throws for non-square matrices or with odd dimensions" begin
    A = randn(4, 5)
    @test_throws ArgumentError premultiply_by_Omega(A)

    B = randn(3, 3)
    @test_throws ArgumentError premultiply_by_Omega(B)
end

@testset "Test premultiply_by_Omega() on identity and symplectic form" begin
    for ell = 1:3
        Omega = [zeros(ell, ell) I(ell); -I(ell) zeros(ell, ell)]
        @test isapprox(premultiply_by_Omega(I(2 * ell)), Omega; atol = 1e-10)
        @test isapprox(premultiply_by_Omega(Omega), -I(2 * ell); atol = 1e-10)
    end
end

@testset "Test successive application of premultiply_by_Omega()" begin
    for ell = 1:3
        A = randn(2 * ell, 2 * ell)
        OOA = premultiply_by_Omega(premultiply_by_Omega(A))
        @test isapprox(OOA, -A; atol = 1e-10)
    end
end

@testset "Test random_symplectic() matrix properties" begin
    for ell = 1:3
        for prec = 2:1:3
            S = random_symplectic(ell; prec = prec)
            @test issymplectic(S) == true
            @test isapprox(det(S), 1.0; atol = 1e-10)
        end
    end
end

@testset "Test takagi() throws for non-square or non-symmetric matrices" begin
    A = randn(4, 5)
    @test_throws ArgumentError takagi(A)

    B = randn(3, 3)
    B = B + transpose(B)
    B[1, 2] = -B[1, 2]
    @test_throws ArgumentError takagi(B)
end

@testset "Test takagi() matrix properties" begin
    for ell = 2:4
        for prec = 2:1:3
            reA = Array{MultiFloat{Float64,prec}}(randn(ell, ell))
            imA = Array{MultiFloat{Float64,prec}}(randn(ell, ell))
            A = reA + im .* imA
            A = (A + transpose(A)) / 2

            D, W = takagi(A)

            @test isapprox(A, W * Diagonal(D) * transpose(W); atol = 1e-10)
            @test isapprox(W' * W, I(ell); atol = 1e-10)
            @test all(D .>= 0)
        end
    end
end

@testset "Test takagi() for real, symmetric, positive-definite matrices" begin
    for ell = 2:4
        for prec = 2:1:3
            A = Array{MultiFloat{Float64,prec}}(randn(ell, ell))
            A = (A + transpose(A)) / 2
            Ap = I(ell) + A * transpose(A)

            D, W = takagi(Ap)
            U, Dp, V = svd(Ap)

            @test isapprox(D, Dp; atol = 1e-10)
            @test isapprox(W, U; atol = 1e-10)
            @test isapprox(W, V; atol = 1e-10)
        end
    end
end

@testset "Test bloch_messiah() throws for non-square, odd dimension, complex or non-symplectic matrices" begin
    A = randn(4, 5)
    @test_throws ArgumentError bloch_messiah(A)

    B = randn(3, 3)
    @test_throws ArgumentError bloch_messiah(B)

    C = randn(3, 3) + im .* randn(3, 3)
    @test_throws ArgumentError bloch_messiah(C)

    D = randn(2, 2)
    D = [D zeros(2, 2); zeros(2, 2) D]
    @test_throws ArgumentError bloch_messiah(D)
end

@testset "Test bloch_messiah() matrix properties" begin
    for ell = 2:4
        for prec = 2:1:3
            S = random_symplectic(ell; prec = prec)

            O1, D, O2 = bloch_messiah(S)

            @test isapprox(S, O1 * Diagonal(D) * O2; atol = 1e-10)
            @test issymplectic(O1) == true
            @test issymplectic(O2) == true
            @test issymplectic(Diagonal(D)) == true
            @test isapprox(O1 * transpose(O1), I(2 * ell); atol = 1e-10)
            @test isapprox(O2 * transpose(O2), I(2 * ell); atol = 1e-10)
        end
    end
end

@testset "Test bloch_messiah() on identity matrix" begin
    for ell = 2:4
        for prec = 2:1:3
            S = I(2 * ell)

            O1, D, O2 = bloch_messiah(S)

            @test isapprox(O1 * O2, I(2 * ell); atol = 1e-10)
            @test isapprox(Diagonal(D), I(2 * ell); atol = 1e-10)
            @test issymplectic(O1) == true
            @test issymplectic(O2) == true
            @test isapprox(O1 * transpose(O1), I(2 * ell); atol = 1e-10)
            @test isapprox(O2 * transpose(O2), I(2 * ell); atol = 1e-10)
        end
    end
end

@testset "Test bloch_messiah() on squeezing" begin
    for ell = 2:4
        for prec = 2:1:3
            r = randn(ell)
            S = Diagonal(vcat(exp.(-r), exp.(r)))

            O1, D, O2 = bloch_messiah(S)

            sD = sort(D)
            sDp = sort(vcat(exp.(-r), exp.(r)))

            @test isapprox(sD, sDp; atol = 1e-10)
            @test isapprox(O1 * O2, I(2 * ell); atol = 1e-10)
            @test issymplectic(O1) == true
            @test issymplectic(O2) == true
            @test isapprox(O1 * transpose(O1), I(2 * ell); atol = 1e-10)
            @test isapprox(O2 * transpose(O2), I(2 * ell); atol = 1e-10)
        end
    end
end

@testset "Test williamson() throws for non-square, odd dimension, complex, non-symmetric or non-positive-definite matrices" begin
    A = randn(4, 5)
    @test_throws ArgumentError williamson(A)

    B = randn(3, 3)
    @test_throws ArgumentError williamson(B)

    C = randn(3, 3) + im .* randn(3, 3)
    @test_throws ArgumentError williamson(C)

    D = randn(4, 4)
    D = D + transpose(D)
    D[1, 2] = -D[1, 2]
    @test_throws ArgumentError williamson(D)

    E = Diagonal(vcat(1.0, 0.0, -1.0, 1.0))
    @test_throws ArgumentError williamson(E)
end

@testset "Test williamson() matrix properties" begin
    for ell = 2:4
        for prec = 2:1:3
            nvec = 2 .* Array{MultiFloat{Float64,prec}}(abs.(randn(ell))) .+ 1
            S1 = random_symplectic(ell; prec = prec)
            Sigma = S1 * Diagonal(vcat(nvec, nvec)) * transpose(S1)
            Sigma = Hermitian(Sigma)

            D, S2 = williamson(Sigma)
            sD = sort(D)
            sDp = sort(vcat(nvec, nvec))

            @test isapprox(Sigma, S2 * Diagonal(D) * transpose(S2); atol = 1e-10)
            @test issymplectic(S2) == true
            @test all(D .> 0)
            @test isapprox(sD, sDp; atol = 1e-10)
        end
    end
end

@testset "Test williamson() on identity" begin
    for ell = 2:4
        for prec = 2:1:3
            D, S = williamson(I(2 * ell))

            @test issymplectic(S) == true
            @test isapprox(S * transpose(S), I(2 * ell); atol = 1e-10)
            @test isapprox(Diagonal(D), I(2 * ell); atol = 1e-10)
        end
    end
end

@testset "Test williamson() on pure states" begin
    for ell = 2:4
        for prec = 2:1:3
            S1 = random_symplectic(ell; prec = prec)
            Sigma = S1 * transpose(S1)

            D, S2 = williamson(Hermitian(Sigma))

            @test issymplectic(S2) == true
            @test isapprox(S2 * transpose(S2), Sigma; atol = 1e-10)
            @test isapprox(Diagonal(D), I(2 * ell); atol = 1e-10)
        end
    end
end

@testset "Test pre_iwasawa() throws for non-square, odd dimension, complex or non-symplectic matrices" begin
    A = randn(4, 5)
    @test_throws ArgumentError pre_iwasawa(A)

    B = randn(3, 3)
    @test_throws ArgumentError pre_iwasawa(B)

    C = randn(3, 3) + im .* randn(3, 3)
    @test_throws ArgumentError pre_iwasawa(C)

    D = randn(2, 2)
    D = [D zeros(2, 2); zeros(2, 2) D]
    @test_throws ArgumentError pre_iwasawa(D)
end

@testset "Test pre_iwasawa() matrix properties" begin
    for ell = 2:4
        for prec = 2:1:3
            S = random_symplectic(ell; prec = prec)
            SS = S * transpose(S)

            E, D, F = pre_iwasawa(S)
            A = D[1:ell, 1:ell]
            CA = E[(ell+1):(2*ell), 1:ell]
            U = F[1:ell, 1:ell] + im .* F[1:ell, (ell+1):(2*ell)]

            @test isapprox(S, E * D * F; atol = 1e-10)
            @test issymplectic(F) == true
            @test isapprox(F * transpose(F), I(2 * ell); atol = 1e-10)
            @test isapprox(A, transpose(A); atol = 1e-10)
            @test isapprox(A * A, SS[1:ell, 1:ell]; atol = 1e-10)
            @test isapprox(CA, transpose(CA); atol = 1e-10)
            @test isapprox(U * U', I(ell); atol = 1e-10)
        end
    end
end

@testset "Test iwasawa() throws for non-square, odd dimension, complex or non-symplectic matrices" begin
    A = randn(4, 5)
    @test_throws ArgumentError iwasawa(A)

    B = randn(3, 3)
    @test_throws ArgumentError iwasawa(B)

    C = randn(3, 3) + im .* randn(3, 3)
    @test_throws ArgumentError iwasawa(C)

    D = randn(2, 2)
    D = [D zeros(2, 2); zeros(2, 2) D]
    @test_throws ArgumentError iwasawa(D)
end

@testset "Test iwasawa() matrix properties" begin
    for ell = 2:4
        for prec = 2:1:3
            S = random_symplectic(ell; prec = prec)
            E, D, F = iwasawa(S)
            U = F[1:ell, 1:ell] + im .* F[1:ell, (ell+1):(2*ell)]
            bD1 = Diagonal(D[1:ell])
            bD2 = Diagonal(D[(ell+1):(2*ell)])
            EA = E[1:ell, 1:ell]
            EB = E[1:ell, (ell+1):(2*ell)]
            EC = E[(ell+1):(2*ell), 1:ell]
            ED = E[(ell+1):(2*ell), (ell+1):(2*ell)]

            @test isapprox(S, E * Diagonal(D) * F; atol = 1e-10)
            @test issymplectic(E) == true
            @test issymplectic(Diagonal(D)) == true
            @test issymplectic(F) == true
            @test isapprox(F * transpose(F), I(2 * ell); atol = 1e-10)
            @test isapprox(U * U', I(ell); atol = 1e-10)
            @test isapprox(bD1 * bD2, I(ell); atol = 1e-10)
            @test isapprox(EB, zeros(ell, ell); atol = 1e-10)
            @test isapprox(transpose(EA) * EC, transpose(EC) * EA; atol = 1e-10)
            @test isapprox(EA * transpose(ED), I(ell); atol = 1e-10)
            @test isapprox(EA, tril(EA); atol = 1e-10)
            @test isapprox(ED, triu(ED); atol = 1e-10)
        end
    end
end

@testset "Test iwasawa() on squeezing" begin
    for ell = 2:4
        for prec = 2:1:3
            r = randn(ell)
            S = Diagonal(vcat(exp.(-r), exp.(r)))

            E, D, F = iwasawa(S)

            @test isapprox(E, I(2 * ell); atol = 1e-10)
            @test isapprox(F, I(2 * ell); atol = 1e-10)
            @test isapprox(Diagonal(D), S; atol = 1e-10)
        end
    end
end

@testset "Test iwasawa() on orthogonal symplectic" begin
    for ell = 2:4
        for prec = 2:1:3
            U = rand(Haar(2), ell)
            rU = Array{MultiFloat{Float64,prec}}(real(U))
            iU = Array{MultiFloat{Float64,prec}}(imag(U))
            S = [rU -iU; iU rU]

            E, D, F = iwasawa(S)

            @test isapprox(E, I(2 * ell); atol = 1e-10)
            @test isapprox(Diagonal(D), I(2 * ell); atol = 1e-10)
            @test isapprox(F, S; atol = 1e-10)
        end
    end
end
