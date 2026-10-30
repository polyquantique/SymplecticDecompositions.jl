using LinearAlgebra
using GenericLinearAlgebra
using Random
using RandomMatrices
using MatrixFactorizations
using MultiFloats
using GenericSchur


"""
    issymplectic(S::AbstractMatrix{T}) where {T}

Checks if a matrix is symplectic.

### Input

- `S` -- Matrix to check.

### Output

- `true` if the matrix is symplectic, `false` otherwise.
"""
function issymplectic(S::AbstractMatrix{T}) where {T}
    n, m = size(S)
    if n != m || n % 2 != 0
        return false
    end

    ell = div(n, 2)
    Omega = [zeros(ell, ell) I(ell); -I(ell) zeros(ell, ell)]
    return isapprox(transpose(S) * Omega * S, Omega; atol=1e-10)
end


"""
    premultiply_by_Omega(S::AbstractMatrix{T}) where {T}

Computes Omega * S, where Omega is the symplectic form.

### Input

- `S` -- Matrix to multiply.

### Output

- The product Omega * S.
"""
function premultiply_by_Omega(S::AbstractMatrix{T}) where {T}
    n, m = size(S)
    if n != m || n % 2 != 0
        throw(ArgumentError("Input matrix must be square and of even dimension."))
    end

    ell = div(n, 2)
    A = S[1:ell, 1:ell]
    B = S[1:ell, ell+1:2*ell]
    C = S[ell+1:2*ell, 1:ell]
    D = S[ell+1:2*ell, ell+1:2*ell]

    return [C D; -A -B]
end


"""
    random_symplectic(m::Int; prec::Int=1)

Generates a random symplectic matrix of shape 2m x 2m with a specified floating-point precision.

### Input

- `m`    -- Number of modes.
- `prec` -- (optional, default: 1) precision for MultiFloat; it will have the form Float64x{prec}.

### Output
- 2m x 2m random symplectic matrix with specified precision.
"""
function random_symplectic(m::Int; prec::Int=1)
    type_precision = MultiFloat{Float64, prec}

    U1, U2 = rand(Haar(2), m), rand(Haar(2), m)
    real_U1, imag_U1 = Array{type_precision}(real(U1)), Array{type_precision}(imag(U1))
    real_U2, imag_U2 = Array{type_precision}(real(U2)), Array{type_precision}(imag(U2))
    
    O = [real_U1 -imag_U1; imag_U1 real_U1]
    P = [real_U2 -imag_U2; imag_U2 real_U2]
    
    r = Array{type_precision}(rand(m))
    sq = Diagonal(vcat(exp.(-r), exp.(r)))
    S = O * sq * P

    return S
end


"""
    takagi(A::AbstractMatrix{T}) where {T}

Computes the Takagi decomposition of a complex symmetric matrix, i.e., A = W * D * W', 
where W is unitary and D is diagonal with non-negative entries.

### Input

- `A` -- Complex symmetric matrix for which to compute the decomposition.

### Output

The Takagi decomposition of the input matrix.
- Vector of non-negative entries of the diagonal matrix D.
- Unitary matrix, W, such that A = W * D * W'.
"""
function takagi(A::AbstractMatrix{T}) where {T}
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Input matrix must be square."))
    end

    if !isapprox(A, transpose(A); atol=1e-10)
        throw(ArgumentError("Input matrix must be symmetric."))
    end

    U, D, V = svd(A)
    pref = U' * conj(V)
    sqrt_pref = sqrt(pref)
    return D, U * sqrt_pref
end


"""
    bloch_messiah(S::AbstractMatrix{T}) where {T}

Computes the Bloch-Messiah (or Euler) decomposition of a real symplectic matrix, i.e., S = L * Gamma * R', 
where L and R are symplectic orthogonal, and Gamma is symplectic diagonal and positive-definite.

### Input

- `S` -- Symplectic matrix for which to compute the decomposition.

### Output

The Bloch-Messiah decomposition of the input matrix.
- Symplectic orthogonal matrix L.
- Vector of entries of the symplectic diagonal matrix Gamma.
- Symplectic orthogonal matrix R.
"""
function bloch_messiah(S::AbstractMatrix{T}) where {T}
    if size(S, 1) != size(S, 2)
        throw(ArgumentError("Input matrix must be square."))
    end

    if !isapprox(imag(S), zeros(size(S)); atol=1e-10)
        throw(ArgumentError("Input matrix must be real."))
    end

    if !issymplectic(S)
        throw(ArgumentError("Input matrix must be symplectic."))
    end

    U, D, V = svd(S)
    P = V * Diagonal(D) * V'
    O = U * V'

    n = size(P, 1)
    ell = div(n, 2)
    
    A = P[1:ell, 1:ell]
    B = P[ell+1:2*ell, 1:ell]
    C = P[ell+1:2*ell, ell+1:2*ell]
    M = A - C + im .* (B + B')

    Lambda, W = takagi(M)
    Lambda = 0.5 .* Lambda
    OW = [real(W) -imag(W); imag(W) real(W)]
    aux_Lamb_term = sqrt.(Lambda .^ 2 .+ 1)
    Gamma = vcat(Lambda + aux_Lamb_term, -Lambda + aux_Lamb_term)

    lO = O * OW
    rO = OW'

    return lO, Gamma, rO
end

"""
    williamson(A::AbstractMatrix{T}) where {T}

Computes the Williamson decomposition of a real, symmmetric and positive-definite matrix,
i.e., A = S * T * S', where S is symplectic and T is diagonal with positive entries.

### Input

- `A` -- Matrix for which to compute the decomposition.

### Output

The Williamson decomposition of the input matrix.
- Symplectic orthogonal matrix S.
- Vector of entries of the diagonal matrix T.
"""
function williamson(A::AbstractMatrix{T}) where {T}
    if size(A, 1) != size(A, 2)
        throw(ArgumentError("Input matrix must be square."))
    end

    if !isapprox(imag(A), zeros(size(A)); atol=1e-10)
        throw(ArgumentError("Input matrix must be real."))
    end

    if !isapprox(A, transpose(A); atol=1e-10)
        throw(ArgumentError("Input matrix must be symmetric."))
    end

    if !isposdef(A)
        throw(ArgumentError("Input matrix must be positive definite."))
    end

    n = size(A, 1)
    ell = div(n, 2)

    sqrt_A = sqrt(Symmetric(A))
    inv_sqrt_A = inv(sqrt_A)
    psi = inv_sqrt_A * premultiply_by_Omega(inv_sqrt_A)
    vals, Op = schur(psi)

    perm = Array(1:2*ell)
    phi = abs.(diag(vals, 1)[[1:2:2*ell;]])
    for i = 1:ell
        if vals[2*i-1, 2*i] <= 0
            perm[2*i-1], perm[2*i] = (perm[2*i], perm[2*i-1])
        end
    end
    perm = vcat(perm[[1:2:2*ell;]], perm[[2:2:2*ell;]])
    phi = vcat(phi, phi)

    O = sqrt_A * Op[:, perm] * Diagonal(sqrt.(phi))
    phi = 1 ./ phi

    return phi, O
end