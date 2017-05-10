using ConstantQ
using Base.Test

import ConstantQ: rawdata, _speckernel, _tempkernel, Frequency
import DSP: hamming, hanning

immutable DummyFrequency <: Frequency
end

let
    freq = DummyFrequency()
    @test_throws Exception nfreqs(freq)
    @test_throws Exception freqs(freq)
end

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
    @test isapprox(q(freq), 34.12708770892056)
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
    d = diff(log.(f))
    for e in d
        @test isapprox(d[1], e)
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

# SpectralKernelMatrix
let
    fs = 16000
    freq = GeometricFrequency(min=60, max=5000)
    kp = KernelProperty(fs, freq, hamming)

    S = spzeros(Complex{Float64}, 5, 5)
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

# TemporalKernelMatrix
let
    fs = 16000
    freq = GeometricFrequency(min=60, max=5000)
    kp = KernelProperty(fs, freq, hamming)

    S = spzeros(Complex{Float64}, 5, 5)
    K = TemporalKernelMatrix(S, kp)
    @test property(K) == kp
    @test rawdata(K) == S
    @test issparse(rawdata(K))
    @test issparse(K)

    S = zeros(Complex{Float64}, 5, 5)
    K = TemporalKernelMatrix(S, kp)
    @test rawdata(K) == S
    @test !issparse(rawdata(K))
    @test !issparse(K)
end

# _speckernel
let
    fs = 16000
    freq = GeometricFrequency(174.5, fs/2)
    K = _speckernel(Float64, fs, freq, hamming, 0.005)
    @test issparse(K)
    @test eltype(K) == Complex{Float64}

    K = _speckernel(Float32, fs, freq, hamming, 0.005)
    @test issparse(K)
    @test eltype(K) == Complex{Float32}
end

# speckernel
let
    fs = 16000
    freq = GeometricFrequency(174.5, fs/2)
    K = speckernel(Float64, fs, freq, hamming, 0.005)
    @test isa(K, SpectralKernelMatrix)
    @test issparse(K)

    K = speckernel(Float64, fs)
    @test isa(K, SpectralKernelMatrix)
    @test issparse(K)

    rawK = ConstantQ.rawdata(K)
    @test size(K) == size(rawK)
    @test length(K) == length(rawK)

    # getindex
    for i=1:10
       @test rawK[i] == K[i]
    end

    @test isa(full(K), DenseMatrix)
end

# _tempkernel
let
    fs = 16000
    freq = GeometricFrequency(174.5, fs/2)
    K = _tempkernel(Float64, fs, freq, hamming)
    @test !issparse(K)
    @test eltype(K) == Complex{Float64}

    K = _tempkernel(Float32, fs, freq, hamming)
    @test !issparse(K)
    @test eltype(K) == Complex{Float32}
end

# tempkernel
let
    fs = 16000
    freq = GeometricFrequency(min=220, max=440)

    # compute speckernel
    K = speckernel(Float64, fs, freq, hamming, 0.0)
    K = rawdata(K)
    fftlen = size(K, 1)

    # back to tempkernel
    k = conj(ifft(full(conj(K .* fftlen)), 1))

    # check correctness
    expected = rawdata(tempkernel(Float64, fs, freq, hamming))
    @test isapprox(k, expected)
end

# cqt user-friendly interface
let
    srand(98765)
    x = rand(Float64, 60700)
    fs = 16000

    freq = GeometricFrequency(min=60, max=5000)
    K = speckernel(Float64, fs, freq)
    X, timeaxis, freqaxis = cqt(x, fs, K)
    @test isa(X, Matrix{Complex{Float64}})

    @test first(timeaxis) == 0.0
    @test last(timeaxis) <= length(x)/fs
    @test first(freqaxis) == 60
    @test last(freqaxis) <= 5000
end

# check correctness
let
    srand(98765)
    x = rand(Float64, 60700)
    fs = 16000
    hopsize = convert(Int, round(fs * 0.01))
    freq = GeometricFrequency(min=220, max=fs/2)

    # CQT in frequency-domain
    # make sure threshold to be zero
    K = speckernel(Float64, fs, freq, hamming, 0.0)
    @test isa(K, SpectralKernelMatrix)
    X1, t1, f1 = cqt(x, fs, K, hopsize=hopsize)

    # CQT in time-domain
    k = tempkernel(Float64, fs, freq, hamming)
    @test isa(k, TemporalKernelMatrix)
    X2, t2, f2 = cqt(x, fs, k, hopsize=hopsize)

    @test isapprox(norm(X1), norm(X2), atol=1.0e-7)
    @test isapprox(X1, X2, atol=1.0e-7)
    @test isapprox(t1, t2)
    @test isapprox(f1, f2)
end
