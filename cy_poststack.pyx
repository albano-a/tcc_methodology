# cy_poststack.pyx
import numpy as np
cimport numpy as np

def run_poststack_loop(np.ndarray[np.float32_t, ndim=3] data_amp,
                       np.ndarray[np.float32_t, ndim=3] imp_background,
                       np.ndarray[np.float32_t, ndim=1] wlet,
                       np.ndarray[np.float32_t, ndim=1] wlet_b,
                       float epsI,
                       float epsR):
    from pylops.avo.poststack import PoststackInversion
    cdef int ix = data_amp.shape[0]
    cdef int iy = data_amp.shape[1]
    cdef int iz = data_amp.shape[2]
    
    sparse_ricker = np.zeros((ix, iy, iz), dtype=np.float32)
    sparse_butter = np.zeros((ix, iy, iz), dtype=np.float32)

    for i in range(ix):
        sparse_ricker[i] = PoststackInversion(
            data_amp[i], wlet / 2, m0=imp_background[i],
            explicit=False, epsI=epsI, epsR=epsR,
            simultaneous=False, iter_lim=1000, damp=0.45)[0]

        sparse_butter[i] = PoststackInversion(
            data_amp[i], wlet_b / 2, m0=imp_background[i],
            explicit=False, epsI=epsI, epsR=epsR,
            simultaneous=False, iter_lim=1000, damp=0.45)[0]

    return sparse_ricker, sparse_butter
