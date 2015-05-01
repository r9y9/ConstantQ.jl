using ConstantQ
using Base.Test

import DSP: hamming

let
    GeometricFrequency(60, 61)
    @test_throws ArgumentError GeometricFrequency(60, 60)
    @test_throws ArgumentError GeometricFrequency(60, 59)
end

let
    f = GeometricFrequency(60, 120, 24.0)
    @test_approx_eq q(f) 34.12708770892056
end

let
    @test nfreqs(GeometricFrequency(60, 120, 20)) == 20
    @test nfreqs(GeometricFrequency(60, 240, 20)) == 40
    @test nfreqs(GeometricFrequency(60, 120, 30)) == 30
    @test nfreqs(GeometricFrequency(60, 240, 30)) == 60
end

# constant diff in log-frequency domain
let
    f = freqs(GeometricFrequency(60, 120))
    d = diff(log(f))
    for e in d
        @test_approx_eq d[1] e
    end
end

# Kernel mat
let
    fs = 16000
    fdef = GeometricFrequency(174.5, fs/2)
    K = kernelmat(Float64, fdef, fs, hamming, 0.005)
    @test issparse(K)
    @test eltype(K) == Complex{Float64}

    K = kernelmat(Float32, fdef, fs, hamming, 0.005)
    @test issparse(K)
    @test eltype(K) == Complex{Float32}
end

# cqt
let
    srand(98765)
    x = rand(Float64, 60700)
    fs = 16000
    fdef = GeometricFrequency(174.5, fs/2)

    K = kernelmat(Float64, fdef, fs, hamming, 0.005)
    X = cqt(x, fs, fdef, hopsize=80, K=K)
    @test isa(X, Matrix{Complex{Float64}})
end

# cqt_naive
let
    srand(98765)
    x = rand(Float64, 60700)
    fs = 16000
    fdef = GeometricFrequency(174.5, fs/2)
    X = ConstantQ.cqt_naive(x, fs, fdef, hopsize=80)
    @test isa(X, Matrix{Complex{Float64}})
end
