import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np
import numpy.ma as ma

import tifffile as tiff

import scipy.ndimage
from scipy.interpolate import interpn

###################################
binning = (2, 2, 2)
true_if_channel = True
###################################
matrix = np.genfromtxt(sys.argv[1])
data = np.squeeze(tiff.imread(sys.argv[2]))
threshold = np.min(data[data > 0]) / 100
cpu = int(cpu_count() / 2)
parallel_bin = cpu
if len(data.shape) < 4:
    data = data[None, ...]
if true_if_channel:
    data = np.swapaxes(data, 0, 1)

x = np.linspace(1, data.shape[2], data.shape[2])
y = np.linspace(1, data.shape[3], data.shape[3])
z = np.linspace(1, data.shape[1], data.shape[1])
new_z = np.linspace(1, data.shape[1], 2.5 * data.shape[1])
ext_z = int(50 - 1.25 * data.shape[1])
grid = np.array(np.meshgrid(new_z, y, x, indexing='ij'))
grid = np.moveaxis(grid, 0, -1)

res = np.zeros((data.shape[0], new_z.size + 2 * ext_z, data.shape[2], data.shape[3]), dtype="float32")


def rotate_instant(datain):
    ti = datain[0]
    interp = interpn((z, y, x), datain[1], grid, 'linear')
    out = scipy.ndimage.affine_transform(interp, matrix, output_shape=res.shape[1:])
    print("transform", ti * 100.0 / data.shape[0])
    return [ti, out.astype("float32")]


data_in = []
for i in range(data.shape[0]):
    data_in.append([i, data[i, :, :, :]])

pool = Pool(processes=cpu)
for tmp in pool.imap(rotate_instant, data_in, chunksize=1):
    res[tmp[0], ...] = tmp[1]
pool.close()
pool.join()

data = None
masked = ma.masked_array(res, mask=res <= threshold)
rescaled = ma.zeros((masked.shape[0], int(masked.shape[1] / binning[0]), int(masked.shape[2] / binning[1]),
                     int(masked.shape[3] / binning[2])))


def bin_instant(datain):
    ti = datain[0]
    m = datain[1]
    out = ma.zeros((m.shape[0], int(masked.shape[1] / binning[0]), int(masked.shape[2] / binning[1]),
                    int(masked.shape[3] / binning[2])), dtype="float32")
    for k in range(rescaled.shape[1]):
        for i in range(rescaled.shape[2]):
            for j in range(rescaled.shape[3]):
                scalata_tmp = m[:, k * binning[0]:(k + 1) * binning[0], i * binning[1]:(i + 1) * binning[1],
                              j * binning[2]:(j + 1) * binning[2]]
                scalata_tmp = scalata_tmp.reshape((m.shape[0], binning[0] * binning[1] * binning[2]))
                out[:, k, i, j] = ma.mean(scalata_tmp, axis=-1)
        print("bin", k)
    return [ti, out]


data_in = []
indexes = np.arange(0, masked.shape[0], parallel_bin).tolist()
indexes.append(masked.shape[0])
for i in range(len(indexes)):
    try:
        data_in.append([[indexes[i], indexes[i + 1]], masked[indexes[i]:indexes[i + 1], :, :, :]])
    except:
        pass

pool = Pool(processes=cpu)
for tmp in pool.imap(bin_instant, data_in, chunksize=1):
    rescaled[tmp[0][0]:tmp[0][1], :, :, :] = tmp[1]
pool.close()
pool.join()

rescaled = ma.filled(rescaled, fill_value=0).astype("float32")

# TZCYXS order.
nome = os.path.basename(sys.argv[2]).split(".")[-2] + "_rotated.tiff"
if true_if_channel:
    rescaled = np.swapaxes(rescaled, 1, 0)
    tiff.imsave(nome, rescaled[None, :, :, :, :], imagej=True)
else:
    tiff.imsave(nome, rescaled[:, :, None, :, :], imagej=True)
