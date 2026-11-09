using LinearAlgebra
using GenericLinearAlgebra
using Random
using RandomMatrices
using GenericSchur
using MultiFloats
MultiFloats.use_bigfloat_transcendentals()


"""
    multi_float_mod(x::T, y::T) where {T<:Real}

Computes the modulus of two real numbers while preserving their floating-point precision.

### Input

- `x` -- First number.
- `y` -- Second number.

### Output

- The modulus of the two numbers.
"""
function multi_float_mod(x::T, y::T) where {T<:Real}
    return x - y * floor(x / y)
end


"""
    multi_float_angle(z::Union{Real, Complex{<:Real}})

Computes the angle of a real or complex number while preserving its floating-point precision.

### Input

- `z` -- Real or complex number for which to compute the angle.

### Output

- The angle of the complex number.
"""
function multi_float_angle(z::Complex{T}) where {T<:Real}
    x, y = real(z), imag(z)
    pi_T = oftype(float(x), pi)

    if iszero(x)
        return iszero(y) ? zero(float(x)) : copysign(pi_T / 2, y)
    end

    theta = atan(y / x)
    if x < 0
        theta += y < 0 ? -pi_T : pi_T
    end
    return theta
end

function multi_float_angle(x::Real)
    x_float = float(x)
    return x < 0 ? oftype(x_float, pi) : zero(x_float)
end


"""
    issymplectic(S::AbstractMatrix{T}) where {T}

Checks if a matrix is symplectic.

### Input

- `S` -- Matrix to check.

### Output

- `true` if the matrix is symplectic, `false` otherwise.
"""
function issymplectic(
    S::AbstractMatrix{T};
    atol::Float64 = 1e-10,
    rtol::Float64 = 0.0,
) where {T}
    n, m = size(S)
    if n != m || n % 2 != 0
        return false
    end

    if !isapprox(imag(S), zeros(size(S)); atol = atol, rtol = rtol)
        throw(ArgumentError("Input matrix must be real."))
    end

    ell = div(n, 2)
    A = S[1:ell, 1:ell]
    B = S[1:ell, (ell+1):(2*ell)]
    C = S[(ell+1):(2*ell), 1:ell]
    D = S[(ell+1):(2*ell), (ell+1):(2*ell)]

    cond1 = isapprox(A' * D - C' * B, I(ell); atol = atol, rtol = rtol)
    cond2 = isapprox(A' * C, C' * A; atol = atol, rtol = rtol)
    cond3 = isapprox(B' * D, D' * B; atol = atol, rtol = rtol)

    return cond1 && cond2 && cond3
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
    B = S[1:ell, (ell+1):(2*ell)]
    C = S[(ell+1):(2*ell), 1:ell]
    D = S[(ell+1):(2*ell), (ell+1):(2*ell)]

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
function random_symplectic(m::Int; prec::Int = 1)
    type_precision = MultiFloat{Float64,prec}

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

The algoritm presented here is adapated from the following thewalrus PR:
https://github.com/XanaduAI/thewalrus/pull/403  

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

    if !isapprox(A, transpose(A); atol = 1e-10)
        throw(ArgumentError("Input matrix must be symmetric."))
    end

    U, D, V = svd(A)
    pref = U' * conj(V)

    #We now compute the square root of pref in a way that avoids jumps from the principal branch
    #of the complex square root function.

    pT, pQ = schur(pref) #pref is normal, so T is diagonal
    pref_eivals = diag(pT)

    #Step 1: Find the unique angles of the eigenvalues of pref, mod 2 * pi, and sort them
    angle_pref_eivals = multi_float_angle.(pref_eivals)
    period = oftype(angle_pref_eivals[1], 2 * pi)
    pref_angles = sort(unique(multi_float_mod.(angle_pref_eivals, period)))

    #Step 2: Find the difference between these angles, including the difference between the first and last angle (mod 2 * pi)
    pref_angles_diff = diff(vcat(pref_angles, pref_angles[1] + period))

    #Step 3: Select the largest difference and find the midpoint of the corresponding arc
    indx_max_diff = argmax(pref_angles_diff)
    mid_point = pref_angles[indx_max_diff] + 0.5 * pref_angles_diff[indx_max_diff]

    #Step 4: Wrap the midpoint around 2 * pi and then shift it by -pi, so it is in the range [-pi, pi]
    shift = multi_float_mod(-mid_point, period) - pi

    #Step 5: Rotate all the eigenvalues so that the midpoint of the largest arc lies at +- pi.
    #In this the other angles will be far from the branch cut of the complex suare root
    shifted_pref_eivals = pref_eivals .* exp.(im * shift)

    #Step 6: Find the square root of the rotated eigenvalues, they should lie in the principal branch.
    #Then compute the square root of the shifted pref.
    sqrt_shifted_pref_eivals = sqrt.(shifted_pref_eivals)
    sqrt_shifted_pref = pQ * Diagonal(sqrt_shifted_pref_eivals) * pQ'

    #Step 7: Undo the rotation by rotating by half the angle in the opposite direction
    sqrt_pref = sqrt_shifted_pref .* exp.(-im * shift / 2)

    return D, U * sqrt_pref
end


"""
    bloch_messiah(S::AbstractMatrix{T}) where {T}

Computes the Bloch-Messiah (or Euler) decomposition of a real symplectic matrix, i.e., S = L * Gamma * R, 
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
    n, m = size(S)
    if n != m || n % 2 != 0
        throw(ArgumentError("Input matrix must be square and of even dimension."))
    end

    if !isapprox(imag(S), zeros(size(S)); atol = 1e-10)
        throw(ArgumentError("Input matrix must be real."))
    end

    if !issymplectic(S)
        throw(ArgumentError("Input matrix must be symplectic."))
    end

    U, D, V = svd(S)
    P = V * Diagonal(D) * V'
    O = U * V'

    ell = div(n, 2)

    A = P[1:ell, 1:ell]
    B = P[(ell+1):(2*ell), 1:ell]
    C = P[(ell+1):(2*ell), (ell+1):(2*ell)]
    M = A - C + im .* (B + B')

    Lambda, W = takagi(Symmetric(M))
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
- Vector of entries of the diagonal matrix T.
- Symplectic orthogonal matrix S.
"""
function williamson(A::AbstractMatrix{T}) where {T}
    n, m = size(A)
    if n != m || n % 2 != 0
        throw(ArgumentError("Input matrix must be square and of even dimension."))
    end

    if !isapprox(imag(A), zeros(size(A)); atol = 1e-10)
        throw(ArgumentError("Input matrix must be real."))
    end

    if !isapprox(A, transpose(A); atol = 1e-10)
        throw(ArgumentError("Input matrix must be symmetric."))
    end

    if !isposdef(A) #isposdef() might be too strict for numerical purposes
        throw(ArgumentError("Input matrix must be positive definite."))
    end

    ell = div(n, 2)

    sqrt_A = sqrt(Symmetric(A))
    inv_sqrt_A = inv(sqrt_A)
    psi = inv_sqrt_A * premultiply_by_Omega(inv_sqrt_A)
    vals, Op = schur(psi)

    perm = Array(1:(2*ell))
    phi = abs.(diag(vals, 1)[[1:2:(2*ell);]])
    for i = 1:ell
        if vals[2*i-1, 2*i] <= 0
            perm[2*i-1], perm[2*i] = (perm[2*i], perm[2*i-1])
        end
    end
    perm = vcat(perm[[1:2:(2*ell);]], perm[[2:2:(2*ell);]])
    phi = vcat(phi, phi)

    O = sqrt_A * Op[:, perm] * Diagonal(sqrt.(phi))
    phi = 1 ./ phi

    return phi, O
end


"""
    pre_iwasawa(S::AbstractMatrix{T}) where {T}

Computes the pre-Iwasawa decomposition of a real symplectic matrix, i.e., S = E * D * F, 
where E and D have the structure

E = ⌈    I       0⌉ , D = ⌈ A0    0    ⌉,
    ⌊ C0*A0^(-1) I⌋       ⌊ 0  A0^(-1) ⌋
    
and F is symplectic orthogonal.

### Input

- `S` -- Symplectic matrix for which to compute the decomposition.

### Output

The pre-Iwasawa decomposition of the input matrix.
- Matrix E.
- Matrix D.
- Symplectic orthogonal matrix F.
"""
function pre_iwasawa(S::AbstractMatrix{T}) where {T}
    n, m = size(S)
    if n != m || n % 2 != 0
        throw(ArgumentError("Input matrix must be square and of even dimension."))
    end

    if !isapprox(imag(S), zeros(size(S)); atol = 1e-10)
        throw(ArgumentError("Input matrix must be real."))
    end

    if !issymplectic(S)
        throw(ArgumentError("Input matrix must be symplectic."))
    end

    ell = div(n, 2)
    A = S[1:ell, 1:ell]
    B = S[1:ell, (ell+1):(2*ell)]
    C = S[(ell+1):(2*ell), 1:ell]
    D = S[(ell+1):(2*ell), (ell+1):(2*ell)]

    A0 = sqrt(Symmetric(A * A' + B * B'))
    inv_A0 = inv(A0)

    X = inv_A0 * A
    Y = inv_A0 * B

    C0 = (C * A' + D * B') * inv_A0

    E = [I(ell) zeros(ell, ell); C0*inv_A0 I(ell)]
    D = [A0 zeros(ell, ell); zeros(ell, ell) inv_A0]
    F = [X Y; -Y X]

    return E, D, F
end


"""
    iwasawa(S::AbstractMatrix{T}) where {T}

Computes the Iwasawa decomposition of a real symplectic matrix, i.e., S = E * D * F, 
where E is nilpotent symplectic, D is symplectic diagonal, and F is symplectic orthogonal.

### Input

- `S` -- Symplectic matrix for which to compute the decomposition.

### Output

The Iwasawa decomposition of the input matrix.
- Nilpotent symplectic matrix E.
- Vector of entries of the symplectic diagonal matrix D.
- Symplectic orthogonal matrix F.
"""
function iwasawa(S::AbstractMatrix{T}) where {T}
    n, m = size(S)
    if n != m || n % 2 != 0
        throw(ArgumentError("Input matrix must be square and of even dimension."))
    end

    if !isapprox(imag(S), zeros(size(S)); atol = 1e-10)
        throw(ArgumentError("Input matrix must be real."))
    end

    if !issymplectic(S)
        throw(ArgumentError("Input matrix must be symplectic."))
    end

    ell = div(n, 2)
    E, D, F = pre_iwasawa(S)

    top_right_block_D = D[1:ell, 1:ell]
    Q, R = qr(top_right_block_D)

    diag_R = diag(R)
    abs_diag_R = abs.(diag_R)
    sign_diag_R = sign.(diag_R)
    R = R' * Diagonal(1 ./ diag_R)
    inv_transpose_R = inv(R)'
    DD = vcat(abs_diag_R, 1 ./ abs_diag_R)
    OO = [R zeros(ell, ell); zeros(ell, ell) inv_transpose_R]

    Q = Diagonal(sign_diag_R) * Q'
    AA = [Q zeros(ell, ell); zeros(ell, ell) Q]

    EE = E * OO
    FF = AA * F

    return EE, DD, FF
end
