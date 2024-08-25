import numpy as np
import gzip
import shutil

amplitude = "F:/Universidade/Dados_Sísmicos/Amplitude/"
falhas = "F:/Universidade/Dados_Sísmicos/Faults/"
impedance = "F:/Universidade/Dados_Sísmicos/Impedance/"


data = np.load(impedance + "impedance_synth.npy")
np.save(impedance + "temp.npy", data)

with open(impedance + "temp.npy", "rb") as f_in:
    with gzip.open(impedance + "data.npy.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

import os

os.remove(impedance + "temp.npy")
