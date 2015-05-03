using DSP

import Base: getindex, size, length, full, issparse

abstract Frequency

nfreqs(f::Frequency) = error("Not implemented")
freqs(f::Frequency) = error("Not implemented")

# Geometrically spaced frequency
# fₖ = min * 2^(1/bins)ᵏ
immutable GeometricFrequency <: Frequency
    min::Real
    max::Real
    bins::Real # the number of bins per octave

    function GeometricFrequency(min, max, bins=24)
        min <= max || throw(ArgumentError("max must be larger than min"))
        bins > 0 || throw(ArgumentError("bins must be positive"))
        new(min, max, bins)
    end
end

function GeometricFrequency(; min::Real=55, max::Real=5000, bins::Real=24)
    GeometricFrequency(min, max, bins)
end

nbins_per_octave(freq::GeometricFrequency) = freq.bins

# Q-factor
q(bins, qrate=1.0) = 1.0 / (2 ^ (1.0/bins) - 1) * qrate
q(f::GeometricFrequency, qrate=1.0) = q(f.bins, qrate)

function nfreqs(f::GeometricFrequency)
    convert(Int, round(nbins_per_octave(f)  * (log2(f.max / f.min))))
end

function freqs(f::GeometricFrequency)
    f.min * 2 .^ ((0:nfreqs(f)-1) / nbins_per_octave(f))
end

# Kernel property
immutable KernelProperty
    fs::Real
    freq::GeometricFrequency
    win::Function

    function KernelProperty(fs, freq, win)
        applicable(win, 512) || error("Invalid window function")
        new(fs, freq, win)
    end
end

# Kernel matrices
abstract KernelMatrix{T} <: AbstractMatrix{T}

# Spectral kernel (freqency-domain kernel)
immutable SpectralKernelMatrix{T<:Complex} <: KernelMatrix{T}
    data::AbstractMatrix{T}
    property::KernelProperty
end

# Temporal kernel
immutable TemporalKernelMatrix{T<:Complex} <: KernelMatrix{T}
    data::AbstractMatrix{T}
    property::KernelProperty
end

property(k::KernelMatrix) = k.property
rawdata(k::KernelMatrix) = k.data

size(k::KernelMatrix) = size(k.data)
length(k::KernelMatrix) = length(k.data)

getindex(k::KernelMatrix, i::Real) = getindex(k.data, i)
getindex(k::KernelMatrix, i::Real...) = getindex(k.data, i...)
getindex(k::KernelMatrix, i::Real, j::Real) = getindex(k.data, i, j)
getindex(k::KernelMatrix, i::Range, j::Real) = getindex(k.data, i, j)
getindex(k::KernelMatrix, i::Real, j::Range) = getindex(k.data, i, j)

issparse(k::KernelMatrix) = issparse(k.data)
full(k::KernelMatrix) = full(k.data)

function tempkernel(T::Type,
                   fs::Real,
                   freq::GeometricFrequency=GeometricFrequency(55, fs/2),
                   win::Function=hamming)
    prop = KernelProperty(fs, freq, win)
    data = _tempkernel(T, prop)
    TemporalKernelMatrix(data, prop)
end

function _tempkernel(T::Type, prop::KernelProperty)
    _tempkernel(T, prop.fs, prop.freq, prop.win)
end

function _tempkernel(T::Type, fs::Real, freq::GeometricFrequency, win::Function)
    Q = q(freq)
    f = freqs(freq)
    winsizes = int(fs ./ f * Q)
    fftlen = nextpow2(winsizes[1])

    K = zeros(Complex{T}, fftlen, length(winsizes))

    for k = 1:length(winsizes)
        Nk = winsizes[k]
        kernel = win(Nk) .* exp(-im*2π*Q/Nk .* (1:Nk)) / Nk
        s = (fftlen - Nk) >> 1
        copy!(K, fftlen*(k-1) + s, kernel, 1, Nk)
    end

    K
end

function speckernel(T::Type,
                   fs::Real,
                   freq::GeometricFrequency=GeometricFrequency(55, fs/2),
                   win::Function=hamming,
                   threshold::Float64=0.005)
    prop = KernelProperty(fs, freq, win)
    data = _speckernel(T, prop, threshold)
    SpectralKernelMatrix(data, prop)
end

function _speckernel(T::Type, prop::KernelProperty, threshold::Float64=0.005)
    _speckernel(T, prop.fs, prop.freq, prop.win, threshold)
end

# Compute sparse kernel matrix in frequency-domain
function _speckernel(T::Type,
                    fs::Real,
                    freq::GeometricFrequency,
                    win::Function,
                    threshold::Float64)
    K = _tempkernel(T, fs, freq, win)
    fftlen = size(K, 1)

    # to frequency domain
    fft!(K, 1)

    # make it sparse
    K[abs(K) .< threshold] = 0.0
    Kˢ = sparse(K)

    # take complex conjugate
    conj!(Kˢ)

    # normalize by fftlen
    Kˢ ./= fftlen

    Kˢ
end

function sym!(symfftout, fftout)
    copy!(symfftout, 1, fftout, 1, length(fftout))
    @inbounds for i = 1:length(fftout)-1
        symfftout[end-i+1] = conj(fftout[i+1])
    end
end

# J. C. Brown and M. S. Puckette, "An efficient algorithm for the calculation
# of a constant Q transform," J. Acoust. Soc. Amer., vol. 92, no. 5,
# pp. 2698–2701, 1992.
function cqt{T}(x::Vector{T},
                fs::Real,
                freq::GeometricFrequency=GeometricFrequency(55, fs/2),
                hopsize::Int=convert(Int, round(fs*0.005)),
                win::Function=hamming,
                K::AbstractMatrix = _speckernel(T, fs, freq, win, 0.005)
                )
    Q = q(freq)
    freqaxis = freqs(freq)

    winsizes = int(fs ./ freqaxis * Q)
    nframes = div(length(x), hopsize)

    fftlen = nextpow2(winsizes[1])
    size(K) == (fftlen, length(winsizes)) || error("inconsistent kernel size")

    # Create padded signal
    xpadd = Array(T, length(x) + fftlen)
    copy!(xpadd, fftlen>>1+1, x, 1, length(x))

    # FFT workspace
    fftin = Array(T, fftlen)
    fftout = Array(Complex{T}, fftlen>>1+1)
    fplan = FFTW.Plan(fftin, fftout, 1, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
    symfftout = Array(Complex{T}, fftlen)

    # Constant-Q spectrogram
    X = Array(Complex{T}, length(freqaxis), nframes)

    # tmp used in loop
    freqbins = Array(Complex{T}, length(freqaxis))

    for n = 1:nframes
        s = hopsize * (n - 1) + 1
        # copy to input buffer
        copy!(fftin, 1, xpadd, s, fftlen)
        # FFT
        FFTW.execute(fplan.plan, fftin, fftout)
        # get full fft bins (rest half are complex conjugate)
        sym!(symfftout, fftout)
        # multiply in frequency-domain
        At_mul_B!(freqbins, K, symfftout)
        # copy to result buffer
        copy!(X, length(freqaxis)*(n-1) + 1, freqbins, 1, length(freqaxis))
    end

    timeaxis = [0:hopsize:nframes;]/fs
    X, timeaxis, freqaxis
end

# CQT with a user-specified kernel matrix
function cqt(x::Vector,
             fs::Real,
             K::SpectralKernelMatrix;
             hopsize::Int = convert(Int, fs * 0.005)
             )
    prop = property(K)
    fs == prop.fs || error("Inconsistent kernel")
    cqt(x, prop.fs, prop.freq, hopsize, prop.win, K.data)
end
