using DSP

abstract Frequency

# Interface
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

GeometricFrequency(fmin, fmax; ratio=24.0) = GeometricFrequency(fmin, fmax, ratio)

# Q-value
q(fratio, qrate=1.0) = 1. / (2 ^ (1/fratio) - 1) * qrate
q(f::GeometricFrequency, qrate=1.0) = q(f.ratio, qrate)

function nfreqs(f::GeometricFrequency)
    round(Int, f.ratio  * (log2(f.max / f.min)))
end

freqs(f::GeometricFrequency) = f.min * 2.0 .^ ((0:nfreqs(f)-1) / f.ratio)

# Compute (sparse) kernel matrix K in frequency-domain
function kernelmat(T::Type, fdef::GeometricFrequency, fs=1,
                   win::Function=hamming,
                   splow::Float64=0.005)
    Q = q(fdef)
    f = freqs(fdef)
    winsizes = round(Int, fs ./ f * Q)
    fftlen = nextpow2(winsizes[1])

    # Create kernel matrix
    K = zeros(Complex{T}, fftlen, length(winsizes))
    tmp = Array(Complex{T}, fftlen)

    for k = 1:length(winsizes)
        fill!(tmp, zero(T))
        Nₖ = winsizes[k]
        kernel = win(Nₖ) .* exp(-im*2π*Q/Nₖ .* (1:Nₖ)) / Nₖ
        s = (fftlen - Nₖ) >> 1 + 1
        copy!(tmp, s, kernel, 1, Nₖ)
        K[:, k] = fft(tmp)
    end

    # approximate to sparse matrix
    K[abs(K) .< splow] = 0.0
    Kˢ = sparse(K)

    # Complex conjugate
    conj!(Kˢ)

    # Normalization
    Kˢ ./= fftlen

    Kˢ
end

function fastcqt{T}(x::Vector{T}, fdef::GeometricFrequency, fs=1;
                    hopsize=round(Int, fs*0.005),
                    win::Function=hamming,
                    splow::Float64=0.0054,
                    K = kernelmat(T, fdef, fs, win, splow)
    )
    Q = q(fdef)
    f = freqs(fdef)

    winsizes = round(Int, fs ./ f * Q)
    nframes = round(Int, length(x) / hopsize) - 1

    fftlen = nextpow2(winsizes[1])
    @assert size(K) == (fftlen, length(winsizes))

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

    for n = 1:nframes
        s = hopsize * n + 1
        # copy to input buffer
        copy!(fftin, 1, xpadd, s, fftlen)
        # FFT
        FFTW.execute(fplan.plan, fftin, fftout)
        # get full fft bins (rest half are complex conjugate)
        copy!(symfftout, 1, fftout, 1, length(fftout))
        @inbounds for i=1:length(fftout)-1
            symfftout[end-i+1] = conj(fftout[i+1])
        end
        # multily in frequency-domain
        X[:,n] = symfftout * K
    end

    X
end

function cqt{T}(x::Vector{T}, f::GeometricFrequency, fs=1;
                hopsize=round(Int, fs*0.005),
                win::Function=hamming)
    Q = q(f)
    f = freqs(f)

    winsizes = round(Int, fs ./ f * Q)
    nframes = round(Int, length(x) / hopsize) - 1

    X = Array(Complex{T}, length(f), nframes)

    for k = 1:length(winsizes)
        Nₖ = winsizes[k]
        Nh = round(Int, Nₖ / 2)
        kernel = win(Nₖ) .* exp(-im*2π*Q/Nₖ .* (1:Nₖ)) / Nₖ
        s = zeros(Nₖ)

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
