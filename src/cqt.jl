using DSP

abstract Frequency

nfreqs(f::Frequency) = error("Not implemented")
freqs(f::Frequency) = error("Not implemented")

# Geometrically spaced frequency
# fₖ = min * 2^(1/ratio)ᵏ
immutable GeometricFrequency <: Frequency
    min::Real
    max::Real
    ratio::Real # the number of bins per octave

    function GeometricFrequency(fmin, fmax, fratio)
        fmin < fmax || throw(ArgumentError("fmax must be larger than fmin"))
        fratio > 0 || throw(ArgumentError("fratio must be positive"))
        new(fmin, fmax, fratio)
    end
end

function GeometricFrequency(fmin, fmax; ratio=24.0)
    GeometricFrequency(fmin, fmax, ratio)
end

# Q-value
q(fratio, qrate=1.0) = 1.0 / (2 ^ (1.0/fratio) - 1) * qrate
q(f::GeometricFrequency, qrate=1.0) = q(f.ratio, qrate)

function nfreqs(f::GeometricFrequency)
    convert(Int, round(f.ratio  * (log2(f.max / f.min))))
end

function freqs(f::GeometricFrequency)
    f.min * 2 .^ ((0:nfreqs(f)-1) / f.ratio)
end

# Compute sparse kernel matrix in frequency-domain
function kernelmat(T::Type, fdef::GeometricFrequency, fs=1,
                   win::Function=hamming,
                   splow::Float64=0.005)
    Q = q(fdef)
    f = freqs(fdef)
    winsizes = int(fs ./ f * Q)
    fftlen = nextpow2(winsizes[1])

    K = zeros(Complex{T}, fftlen, length(winsizes))
    atom = Array(Complex{T}, fftlen)

    for k = 1:length(winsizes)
        fill!(atom, zero(T))
        Nk = winsizes[k]
        kernel = win(Nk) .* exp(-im*2π*Q/Nk .* (1:Nk)) / Nk
        s = (fftlen - Nk) >> 1 + 1
        copy!(atom, s, kernel, 1, Nk)
        K[:, k] = fft(atom)
    end

    K[abs(K) .< splow] = 0.0
    Kˢ = sparse(K)
    conj!(Kˢ)
    Kˢ ./= fftlen

    Kˢ
end

function sym!(symfftout, fftout)
    copy!(symfftout, 1, fftout, 1, length(fftout))
    @inbounds for i = 1:length(fftout)-1
        symfftout[end-i+1] = conj(fftout[i+1])
    end
end

# J. C. Brown and M. S. Puckette, BAn efficient algorithm for the calculation
# of a constant Q transform, J. Acoust. Soc. Amer., vol. 92, no. 5,
# pp. 2698–2701, 1992.
function cqt{T}(x::Vector{T}, fdef::GeometricFrequency, fs=1;
                hopsize::Int=convert(Int, round(fs*0.005)),
                win::Function=hamming,
                splow::Float64=0.0054,
                K::SparseMatrixCSC = kernelmat(T, fdef, fs, win, splow)
    )
    Q = q(fdef)
    f = freqs(fdef)

    winsizes = int(fs ./ f * Q)
    nframes = div(length(x), hopsize) - 1

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
    X = Array(Complex{T}, length(f), nframes)

    # tmp used in loop
    freqbins = Array(Complex{T}, length(f))

    for n = 1:nframes
        s = hopsize * n + 1
        # copy to input buffer
        copy!(fftin, 1, xpadd, s, fftlen)
        # FFT
        FFTW.execute(fplan.plan, fftin, fftout)
        # get full fft bins (rest half are complex conjugate)
        sym!(symfftout, fftout)
        # mutiply in frequency-domain
        At_mul_B!(freqbins, K, symfftout)
        copy!(X, length(f)*(n-1) + 1, freqbins, 1, length(f))
    end

    X
end

# J. Brown. Calculation of a constant Q spectral transform.
# Journal of the Acoustical Society of America,, 89(1):
# 425–434, 1991.
function cqt_naive{T}(x::Vector{T}, f::GeometricFrequency, fs=1;
                      hopsize::Int=convert(Int, round(fs*0.005)),
                      win::Function=hamming)
    Q = q(f)
    f = freqs(f)

    winsizes = int(fs ./ f * Q)
    nframes = div(length(x), hopsize) - 1

    X = Array(Complex{T}, length(f), nframes)

    for k = 1:length(winsizes)
        Nk = winsizes[k]
        Nh = Nk >> 1
        kernel = win(Nk) .* exp(-im*2π*Q/Nk .* (1:Nk)) / Nk
        s = zeros(Nk)

        for n = 1:nframes
            center = hopsize * (n - 1) + 1
            left = max(1, center - Nh)
            copy!(s, 1, x, left, center - left)
            right = min(length(x), center + Nh)
            copy!(s, Nh, x, center, right - center)
            X[k,n] = dot(s, kernel)
        end
    end

    X
end
