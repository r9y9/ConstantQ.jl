using SparseArrays
using DSP, FFTW

import Base: getindex, size, length
import LinearAlgebra: mul!

@compat abstract type Frequency end

nfreqs(f::Frequency) = error("Not implemented")
freqs(f::Frequency) = error("Not implemented")

# Geometrically spaced frequency
# fₖ = min * 2^(1/bins)ᵏ
struct GeometricFrequency <: Frequency
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
q(bins, qrate=1.0) = 1.0 / (2^(1.0 / bins) - 1) * qrate
q(f::GeometricFrequency, qrate=1.0) = q(f.bins, qrate)

function nfreqs(f::GeometricFrequency)
    convert(Int, round(nbins_per_octave(f) * (log2(f.max / f.min))))
end

function freqs(f::GeometricFrequency)
    f.min * 2 .^ ((0:nfreqs(f)-1) / nbins_per_octave(f))
end

# Kernel property
struct KernelProperty
    fs::Real
    freq::GeometricFrequency
    win::Function

    function KernelProperty(fs, freq, win)
        applicable(win, 512) || error("Invalid window function")
        new(fs, freq, win)
    end
end

# Kernel matrices
@compat abstract type KernelMatrix{T} <: AbstractMatrix{T} end

# Spectral kernel (freqency-domain kernel)
struct SpectralKernelMatrix{T<:Complex} <: KernelMatrix{T}
    data::AbstractMatrix{T}
    property::KernelProperty
end

# Temporal kernel
struct TemporalKernelMatrix{T<:Complex} <: KernelMatrix{T}
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
getindex(k::KernelMatrix, i::AbstractRange, j::Real) = getindex(k.data, i, j)
getindex(k::KernelMatrix, i::Real, j::AbstractRange) = getindex(k.data, i, j)

issparse(k::KernelMatrix) = issparse(k.data)
full(k::KernelMatrix) = full(k.data)

nextpow2(n::Real) = (2^(ceil(Int64, log2(n))))

function tempkernel(T::Type,
    fs::Real,
    freq::GeometricFrequency=GeometricFrequency(55, fs / 2),
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
    winsizes = map(Int, map(trunc, (fs ./ f * Q)))
    fftlen = nextpow2(winsizes[1])

    K = zeros(Complex{T}, fftlen, length(winsizes))

    for k = 1:length(winsizes)
        Nk = winsizes[k]
        kernel = win(Nk) .* exp.(-im * 2π * Q / Nk .* (1:Nk)) / Nk
        s = (fftlen - Nk) >> 1
        copyto!(K, fftlen * (k - 1) + s, kernel, 1, Nk)
    end

    K
end

function speckernel(T::Type,
    fs::Real,
    freq::GeometricFrequency=GeometricFrequency(55, fs / 2),
    win::Function=hamming,
    threshold=0.005)
    prop = KernelProperty(fs, freq, win)
    data = _speckernel(T, prop, threshold)
    SpectralKernelMatrix(data, prop)
end

function _speckernel(T::Type, prop::KernelProperty, threshold=0.005)
    _speckernel(T, prop.fs, prop.freq, prop.win, threshold)
end

# Compute sparse kernel matrix in frequency-domain
function _speckernel(T::Type,
    fs::Real,
    freq::GeometricFrequency,
    win::Function,
    threshold)
    K = _tempkernel(T, fs, freq, win)
    conj!(K)

    fftlen = size(K, 1)

    # to frequency domain
    fft!(K, 1)

    # make it sparse
    K[abs.(K).<threshold] .= 0.0
    Kˢ = sparse(K)

    # take complex conjugate
    conj!(Kˢ)

    # normalize by fftlen
    Kˢ ./= fftlen

    Kˢ
end

function sym!(symfftout, fftout)
    copyto!(symfftout, 1, fftout, 1, length(fftout))
    @inbounds for i = 1:length(fftout)-1
        symfftout[end-i+1] = conj(fftout[i+1])
    end
end

# J. C. Brown and M. S. Puckette, "An efficient algorithm for the calculation
# of a constant Q transform," J. Acoust. Soc. Amer., vol. 92, no. 5,
# pp. 2698–2701, 1992.
function cqt(x::Vector{T},
    fs::Real,
    freq::GeometricFrequency=GeometricFrequency(55, fs / 2),
    hopsize::Int=convert(Int, round(fs * 0.005)),
    win::Function=hamming,
    K::AbstractMatrix=_speckernel(T, fs, freq, win, 0.005)
) where {T}
    Q = q(freq)
    freqaxis = freqs(freq)

    winsizes = map(Int, map(trunc, (fs ./ freqaxis * Q)))
    nframes = div(length(x), hopsize)

    fftlen = nextpow2(winsizes[1])
    size(K) == (fftlen, length(winsizes)) || error("inconsistent kernel size")

    # Create padded signal
    xpadd = zeros(T, length(x) + fftlen)
    copyto!(xpadd, fftlen >> 1 + 1, x, 1, length(x))

    # FFT workspace
    fftin = Vector{T}(undef, fftlen)
    fftout = Vector{Complex{T}}(undef, fftlen >> 1 + 1)
    fplan = plan_rfft(fftin)
    symfftout = Vector{Complex{T}}(undef, fftlen)

    # Constant-Q spectrogram
    X = Array{Complex{T},2}(undef, length(freqaxis), nframes)

    # tmp used in loop
    freqbins = Vector{Complex{T}}(undef, length(freqaxis))

    for n = 1:nframes
        s = hopsize * (n - 1) + 1
        # copy to input buffer
        copyto!(fftin, 1, xpadd, s, fftlen)
        # FFT
        mul!(fftout, fplan, fftin)
        # get full fft bins (rest half are complex conjugate)
        sym!(symfftout, fftout)
        # multiply in frequency-domain
        mul!(freqbins, transpose(K), symfftout)
        # copy to output buffer
        copyto!(X, length(freqaxis) * (n - 1) + 1, freqbins, 1, length(freqaxis))
    end

    timeaxis = [0:hopsize:nframes;] / fs
    X, timeaxis, freqaxis
end

# CQT with frequency-domain kernel matrix
function cqt(x::Vector,
    fs::Real,
    K::SpectralKernelMatrix;
    hopsize::Int=convert(Int, round(fs * 0.005))
)
    prop = property(K)
    fs == prop.fs || error("Inconsistent kernel")
    cqt(x, prop.fs, prop.freq, hopsize, prop.win, K.data)
end

# J. C. Brown. "Calculation of a constant Q spectral transform,"
# Journal of the Acoustical Society of America,, 89(1):
# 425–434, 1991.
function time_domain_cqt(x::Vector{T} where {T},
    fs::Real,
    freq::GeometricFrequency=GeometricFrequency(55, fs / 2),
    hopsize::Int=convert(Int, round(fs * 0.005)),
    win::Function=hamming,
    K::AbstractMatrix=_tempkernel(T, fs, freq, win) where {T}
)
    Q = q(freq)
    freqaxis = freqs(freq)

    winsizes = map(Int, map(trunc, (fs ./ freqaxis * Q)))
    nframes = div(length(x), hopsize)

    fftlen = nextpow2(winsizes[1])
    size(K) == (fftlen, length(winsizes)) || error("inconsistent kernel size")

    # Create padded signal
    xpadd = zeros(T, length(x) + fftlen)
    copy!(xpadd, fftlen >> 1 + 1, x, 1, length(x))

    # Constant-q spectrogram
    X = Array{Complex{T},2}(length(freqaxis), nframes)

    # buffer used in roop
    block = Vector{T}(fftlen)
    freqbins = Vector{Complex{T}}(size(X, 1))

    for n = 1:nframes
        s = hopsize * (n - 1) + 1
        # copy to input buffer
        copy!(block, 1, xpadd, s, fftlen)
        # multiply in time-domain
        At_mul_B!(freqbins, K, block)
        # copy to output buffer
        copy!(X, length(freqaxis) * (n - 1) + 1, freqbins, 1, length(freqaxis))
    end

    timeaxis = [0:hopsize:nframes;] / fs
    X, timeaxis, freqaxis
end

# CQT with a time-domain kernel matrix
function cqt(x::Vector,
    fs::Real,
    K::TemporalKernelMatrix;
    hopsize::Int=convert(Int, round(fs * 0.005))
)
    prop = property(K)
    fs == prop.fs || error("Inconsistent kernel")
    time_domain_cqt(x, fs, prop.freq, hopsize, prop.win, rawdata(K))
end
