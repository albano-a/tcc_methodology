import numpy as np
import pylops
from pylops.optimization.sparsity import irls
from tqdm import tqdm
from joblib import Parallel, delayed

def process_trace(ix, iy, amplitude, imp_back, PPop, PPop_b, TIKHO, wlet, wlet_b):
    data_amp = amplitude[ix, iy, :].flatten()
    imp_background = imp_back[ix, iy, :].flatten()

    sparse_spike_inv_ricker = pylops.avo.poststack.PoststackInversion(
        data_amp,#/10,
        wlet / 2,
        m0=imp_background,
        explicit=True,
        epsR=[0.02],
        epsRL1=[0.02],
        simultaneous = True,
        **dict(iter_lim=100, damp=0.2)
    )[0]

    sparse_spike_inv_butter = pylops.avo.poststack.PoststackInversion(
        data_amp,#/10,
        wlet_b / 2,
        m0=imp_background,
        explicit=True,
        epsR=[0.02],
        epsRL1=[0.02],
        simultaneous = True,
        **dict(iter_lim=100, damp=0.2)
    )[0]

    return (ix, iy, sparse_spike_inv_ricker, sparse_spike_inv_butter)

def sparse_spike_inverting(amplitude, imp_back, impedance, PPop, PPop_b, TIKHO):
    nx, ny, nz = amplitude.shape

    inv_imp_ricker = np.zeros((nx, 1, nz))
    inv_imp_butter = np.zeros((nx, 1, nz))

    results = Parallel(n_jobs=-1)(
        delayed(process_trace)(ix, 128, amplitude, imp_back, PPop, PPop_b, TIKHO)
        for ix in tqdm(range(nx), desc="Processing traces")
    )

    for ix, _, sparse_spike_inv_ricker, sparse_spike_inv_butter in results:
        inv_imp_ricker[ix, 0, :] = sparse_spike_inv_ricker.reshape(nz)
        inv_imp_butter[ix, 0, :] = sparse_spike_inv_butter.reshape(nz)

    inverted_impedance_final_ricker = inv_imp_ricker #+ imp_back
    inverted_impedance_final_butter = inv_imp_butter #+ imp_back

    return inverted_impedance_final_ricker, inverted_impedance_final_butter


# Example usage
ss_inverted_impedance_final_ricker2, ss_inverted_impedance_final_butter2 = (
    sparse_spike_inverting(
        data_amp, imp_background, data_imp, poststack_ricker, poststack_butter, TIKHO, wlet, wlet_b
    )
)
np.max(ss_inverted_impedance_final_ricker)