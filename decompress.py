import gzip
import shutil


with gzip.open("Data/amplitude.npy.gz", "rb") as f_in:
    with open("Uncompressed/amplitude.npy", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open("Data/faults.npy.gz", "rb") as f_in:
    with open("Uncompressed/faults.npy", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

with gzip.open("Data/impedance.npy.gz", "rb") as f_in:
    with open("Uncompressed/impedance.npy", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
