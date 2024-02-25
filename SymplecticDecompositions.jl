using LinearAlgebra
using Random
using RandomMatrices
using MatrixFactorizations

"""
    random_symplectic(N)

Generate a random symplectic matrix.

### Input
- `N`  -- Int: Number of modes
### Output

Array: ``2N \\times 2N`` random symplectic matrix.
"""
function random_symplectic(N)
    U = rand(Haar(2), N)
    O = [real(U) -imag(U); imag(U) real(U)]
    U = rand(Haar(2), N)
    P = [real(U) -imag(U); imag(U) real(U)]
    r = abs.(randn(N))
    sq = diagm(vcat(exp.(-r), exp.(r)))
    S = O * sq * P
    return S
end


function normal_sqrtm(A)
    T, Z = schur(A)
    return Z * sqrt(T) * transpose(conj(Z))
end

"""
    takagi(M)

Implements Autonne-Takagi decomposition of a symmetric matrix

### Input
- `M`  -- Array: Symmetric matrix
### Output

Array: ``2N \\times 2N`` random symplectic matrix.
"""
function takagi(M)
    u, s, v = svd(M)
    pref = u' * conj(v)
    pref12 = normal_sqrtm(pref)
    return s, u * pref12
end

function bloch_messiah(S)
    u, d, v = svd(S)
    P = v * diagm(d) * v'
    O = u * v'
    n, m = size(P)
    ell = div(n, 2)
    A = P[1:ell, 1:ell]
    B = P[ell+1:2*ell, 1:ell]
    C = P[ell+1:2*ell, ell+1:2*ell]
    M = A - C + im * (B + B')
    Lam, W = takagi(M)
    Lam = 0.5 * Lam
    OO = [real(W) -imag(W); imag(W) real(W)]
    sqrt1pLam2 = sqrt.(Lam .^ 2 .+ 1)
    D = vcat(Lam + sqrt1pLam2, -Lam + sqrt1pLam2)
    lO = O * OO
    rO = OO'
    return lO, D, rO
end

function williamson(SS)
    n, m = size(S)
    ell = div(n, 2)
    sqrtSS = sqrt(Symmetric(SS))
    sqrtinvSS = inv(sqrtSS)
    psi = sqrtinvSS * premultiplyOmega(sqrtinvSS)
    vals, Otilde = schur(psi)
    perm = Array(1:2*ell)
    for i = 1:ell
        if vals[2*i-1, 2*i] <= 0
            perm[2*i-1], perm[2*i] = (perm[2*i], perm[2*i-1])
        end
    end

    phi = abs.(diag(vals, 1)[[1:2:2*ell;]])
    perm = vcat(perm[[1:2:2*ell;]], perm[[2:2:2*ell;]])
    phi = vcat(phi, phi)
    O = sqrtSS * Otilde[:, perm] * diagm(sqrt.(phi))
    phi = 1 ./ phi
    return phi, O
end

function sympmat(ell)
    idell = Matrix(1.0I, ell, ell)
    zeroell = 0 * idell
    return [zeroell idell; -idell zeroell]
end

function premultiplyOmega(S)
    n, m = size(S)
    ell = div(n, 2)
    A = S[1:ell, 1:ell]
    B = S[1:ell, ell+1:2*ell]
    C = S[ell+1:2*ell, 1:ell]
    D = S[ell+1:2*ell, ell+1:2*ell]
    return [C D; -A -B]
end

function is_symplectic(S)
    n, m = size(S)
    if n % 2 != 0 || m % 2 != 0
        return false
    end
    n = div(n, 2)
    omega = sympmat(n)
    if isapprox(S * premultiplyOmega(S'), omega)
        return true
    end
    return false
end


function pre_iwasawa(S)
    n, m = size(S)
    ell = div(n, 2)
    A = S[1:ell, 1:ell]
    B = S[1:ell, ell+1:2*ell]
    C = S[ell+1:2*ell, 1:ell]
    D = S[ell+1:2*ell, ell+1:2*ell]
    A0 = sqrt(Symmetric(A * A' + B * B'))
    A0inv = inv(A0)
    X = A0inv * A
    Y = A0inv * B
    C0 = (C * A' + D * B') * A0inv
    idell = Matrix(1.0I, ell, ell)
    zeroell = 0 * idell
    E = [idell zeroell; C0*A0inv idell]
    D = [A0 zeroell; zeroell A0inv]
    F = [X Y; -Y X]
    return E, D, F
end

function iwasawa(S)
    n, m = size(S)
    ell = div(n, 2)
    E, D, F = pre_iwasawa(S)
    DNN = D[1:ell, 1:ell]
    Q, R = qr(DNN)
    R = R'
    Q = Q'
    dR = diag(R)
    dd = abs.(dR)
    ds = sign.(dR)
    R = R * diagm(1 ./ dR)
    RinvT = inv(R)'
    DD = vcat(dd, 1 ./ dd)
    zeroell = Matrix(0.0I, ell, ell)
    OO = [R zeroell; zeroell RinvT]
    Q = diagm(ds) * Q
    AA = [Q zeroell; zeroell Q]
    EE = E * OO
    FF = AA * F
    return EE, DD, FF
end
