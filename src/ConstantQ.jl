module ConstantQ

export
    GeometricFrequency,   # geometrically spaced frequency
    KernelProperty,       # kernel property
    SpectralKernelMatrix, # frequency-domain kernel matrix
    property,
    nbins_per_octave,     # number of frequency bins per octave
    nfreqs,               # number of frequency bins
    freqs,                # generate array of frequencies
    q,                    # Q-factor
    kernelmat,            # construct kernel matrix
    cqt                   # A fast constant-Q transform

include("cqt.jl")

end # module
