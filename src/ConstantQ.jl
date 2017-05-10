__precompile__()
module ConstantQ

using Compat

export
    GeometricFrequency,   # geometrically spaced frequency
    KernelProperty,       # kernel property
    SpectralKernelMatrix, # frequency-domain kernel matrix
    TemporalKernelMatrix, # time-domain kernel matrix
    property,
    nbins_per_octave,     # number of frequency bins per octave
    nfreqs,               # number of frequency bins
    freqs,                # generate array of frequencies
    q,                    # Q-factor
    speckernel,           # construct frequency-domain kernel
    tempkernel,           # construct time-domain kernel
    cqt                   # A fast constant-Q transform

include("cqt.jl")

end # module
