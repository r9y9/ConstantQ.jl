using ConstantQ
using Base.Test

import ConstantQ: rawdata, _kernelmat
import DSP: hamming

let
    GeometricFrequency(60, 61)

    # keyword arguments
    GeometricFrequency(min=60, max=5000)
    GeometricFrequency(min=60, max=5000, bins=24)
end

let
    GeometricFrequency(60, 61)
    GeometricFrequency(60, 60)
    @test_throws ArgumentError GeometricFrequency(60, 59)

    GeometricFrequency(60, 5000, 24)
    @test_throws ArgumentError GeometricFrequency(60, 5000, -1)
end

let
    for bins in [24, 48]
        freq = GeometricFrequency(min=55, max=2000, bins=bins)
        @test nbins_per_octave(freq) == bins
    end
end

let
    freq = GeometricFrequency(min=60, max=5000, bins=24)
    @test_approx_eq q(freq) 34.12708770892056
end

let
    @test nfreqs(GeometricFrequency(60, 120, 20)) == 20
    @test nfreqs(GeometricFrequency(60, 240, 20)) == 40
    @test nfreqs(GeometricFrequency(60, 120, 30)) == 30
    @test nfreqs(GeometricFrequency(60, 240, 30)) == 60
end

# constant diff in log-frequency domain
let
    f = freqs(GeometricFrequency(min=60, max=5000))
    d = diff(log(f))
    for e in d
        @test_approx_eq d[1] e
    end
end

function invalid_win()
    println("invalid")
end

let
    fs = 16000
    freq = GeometricFrequency(min=60, max=5000)
    KernelProperty(fs, freq, hamming)
    @test_throws Exception KernelProperty(fs, freq, invalid_win)
end

let
    fs = 16000
    freq = GeometricFrequency(min=60, max=5000)
    kp = KernelProperty(fs, freq, hamming)

    S = spzeros(Complex{Float64}, 5, 5)
    @show typeof(S), typeof(kp)
    K = SpectralKernelMatrix(S, kp)
    @test property(K) == kp
    @test rawdata(K) == S
    @test issparse(rawdata(K))
    @test issparse(K)

    S = zeros(Complex{Float64}, 5, 5)
    K = SpectralKernelMatrix(S, kp)
    @test rawdata(K) == S
    @test !issparse(rawdata(K))
    @test !issparse(K)
end

# _kernelmat
let
    fs = 16000
    freq = GeometricFrequency(174.5, fs/2)
    K = _kernelmat(Float64, fs, freq, hamming, 0.005)
    @test issparse(K)
    @test eltype(K) == Complex{Float64}

    K = _kernelmat(Float32, fs, freq, hamming, 0.005)
    @test issparse(K)
    @test eltype(K) == Complex{Float32}
end

# kernelmat
let
    fs = 16000
    freq = GeometricFrequency(174.5, fs/2)
    K = kernelmat(Float64, fs, freq, hamming, 0.005)
    @test isa(K, SpectralKernelMatrix)
    @test issparse(K)

    K = kernelmat(Float64, fs)
    @test isa(K, SpectralKernelMatrix)
    @test issparse(K)
end

# cqt
# TODO(ryuichi) add tests that checks cqt work correctly
let
    srand(98765)
    x = rand(Float64, 60700)
    fs = 16000
    hopsize = int(fs * 0.001)
    freq = GeometricFrequency(174.5, fs/2)

    K = _kernelmat(Float64, fs, freq, hamming, 0.005)
    X = cqt(x, fs, freq, hopsize, hamming, K)
    @test isa(X, Matrix{Complex{Float64}})

    # user-friendly interface
    K = kernelmat(Float64, fs, freq, hamming, 0.005)
    @test isa(K, SpectralKernelMatrix)
    X = cqt(x, fs, K)
    @test isa(X, Matrix{Complex{Float64}})
end

let
    srand(98765)
    x = rand(Float64, 60700)
    fs = 16000

    K = kernelmat(Float64, fs)
    X = cqt(x, fs, K)
    @test isa(X, Matrix{Complex{Float64}})
end

# cqt_naive
let
    srand(98765)
    x = rand(Float64, 60700)
    fs = 16000
    freq = GeometricFrequency(174.5, fs/2)
    X = ConstantQ.cqt_naive(x, fs, freq, 80)
    @test isa(X, Matrix{Complex{Float64}})
end
