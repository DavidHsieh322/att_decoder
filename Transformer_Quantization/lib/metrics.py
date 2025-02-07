def compute_reconstruction_error(v, vr, mean: bool = True):
    """
    Inputs:
        v: original vectors, shape: (N, d)
        vr: reconstructed vectors, shape: (N, d)
        mean: average the reconstruction error
    """
    recon_err = ((v - vr)**2).sum(axis=1)
    if mean:
        recon_err = recon_err.mean()

    return recon_err