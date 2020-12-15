import os
import argparse
import numpy as np
import pygmo as pg
import scipy.ndimage
from scipy.interpolate import interpn
import tifffile as tiff
from scipy.signal import medfilt2d


class AlignmentProblem:
    def __init__(self, input_file1, input_file2):
        self.T = np.zeros(4)
        self.T[-1] = 1
        self.shear = np.zeros(3)
        self.angles = np.zeros(3)
        self.zoom = np.ones(4)
        self.input_file1 = input_file1
        self.input_file2 = input_file2
        self.output_file = None
        self.a2_min = None
        self.centerT = None
        self.ext_z = None

        self._load_files()

    def _load_files(self):
        self.a1 = tiff.imread(self.input_file1)
        for i0 in range(self.a1.shape[0]):
            self.a1[i0, :, :] = medfilt2d(self.a1[i0, :, :], 3)
        self.a2 = tiff.imread(self.input_file2)
        for i0 in range(self.a2.shape[0]):
            self.a2[i0, :, :] = medfilt2d(self.a2[i0, :, :], 3)
        self.centerT = np.r_[np.array(self.a2.shape) / 2, 1]

        self._interpolate()
        self.a1 = self.a1 > 0
        self.a2_min = np.min(self.a2[self.a2 > 0])

    def _interpolate(self):
        x = np.linspace(1, self.a1.shape[2], self.a1.shape[2])
        y = np.linspace(1, self.a1.shape[1], self.a1.shape[1])
        z = np.linspace(1, self.a1.shape[0], self.a1.shape[0])
        new_z = np.linspace(1, self.a1.shape[0], 2.5 * self.a1.shape[0])
        self.ext_z = int(50-1.25 * self.a1.shape[0])

        grid = np.array(np.meshgrid(new_z, y, x, indexing='ij'))
        grid = np.moveaxis(grid, 0, -1)

        self.a1 = interpn((z, y, x), self.a1, grid, 'linear')
        self.a2 = interpn((z, y, x), self.a2, grid, 'linear')

        self.a1 = np.pad(self.a1, ((self.ext_z, self.ext_z), (0, 0), (0, 0)), mode='constant')

    @property
    def params(self):
        return np.r_[self.T[:3], self.angles, self.zoom[:3], self.shear]

    @params.setter
    def params(self, x):
        self.T[:3] = x[:3]
        self.angles = x[3:6]
        self.zoom[:3] = x[6:9]
        self.shear = x[9:]

    @property
    def affine_matrix(self):
        c = np.cos(self.angles)
        s = np.sin(self.angles)

        R0 = np.array([[c[0], s[0], 0], [-s[0], c[0], 0], [0, 0, 1]])
        R1 = np.array([[c[1], 0, -s[1]], [0, 1, 0], [s[1], 0, c[1]]])
        R2 = np.array([[1, 0, 0], [0, c[2], s[2]], [0, -s[2], c[2]]])

        R = np.linalg.multi_dot([R0, R1, R2])

        tempR = np.zeros((4, 4))
        tempR[:3, :3] = R
        tempR[-1, -1] = 1
        tempT = np.eye(4, 4)
        tempT[:, -1] = self.centerT
        tempT2 = np.copy(tempT)
        tempT2[:3, -1] *= -1

        R = np.linalg.multi_dot([tempT, tempR, tempT2])  # rotation relative to center

        Z = self.zoom * np.eye(4)
        Z[-1, -1] = 1
        Z = np.linalg.multi_dot([tempT, Z, tempT2])  # zoom relative to center

        S = np.eye(4)
        S[0, 1] = self.shear[0]  # shear_yz
        S[0, 2] = self.shear[1]  # shear_xz
        S[1, 2] = self.shear[2]  # shear_xy
        S = np.linalg.multi_dot([tempT, S, tempT2])  # shear relative to center

        M = np.linalg.multi_dot([R, Z, S])
        M[:3, -1] += self.T[:3]

        return M

    @staticmethod
    def get_bounds():
        a = np.array([
            [-25, 25],     # Tz
            [-100, 100],   # Ty
            [-100, 100],   # Tx
            [-np.pi / 6, np.pi / 6],     # theta1
            [-np.pi / 6, np.pi / 6],     # theta2
            [-np.pi / 6, np.pi / 6],     # theta3
            [0.66, 1.5],     # zoom
            [0.66, 1.5],     # zoom
            [0.66, 1.5],     # zoom
            [0, 0],   # shear_yz
            [0, 0],   # shear_xz
            [0, 0],   # shear_xy
        ])

        bounds = (a[:, 0], a[:, 1])
        return bounds

    def fitness(self, x):
        self.params = x
        a2 = scipy.ndimage.affine_transform(
            self.a2, self.affine_matrix, output_shape=(self.a2.shape[0] + 2 * self.ext_z,) + self.a2.shape[1:])
        a2[a2 < self.a2_min / 10.0] = 0

        f = [float(np.sum(np.logical_xor(self.a1, a2)))]

        return f


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test affine aligment',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file1', type=str)
    parser.add_argument('input_file2', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()
    return args


def main():
    nThread=12
    nTemp=10
    tMax=30.0
    args = parse_args()
    prob = AlignmentProblem(args.input_file1, args.input_file2)

    archi = pg.archipelago(n=0)

    pop = pg.population(prob=prob, size=5)
    for i in range(0, nThread):
        # each algo gets a different seed
        algo = pg.simulated_annealing(Ts=tMax, Tf=1, n_T_adj=nTemp)
        # algo = pg.compass_search(max_fevals=50)
        # algo = pg.ihs(gen=20)
        # algo = pg.de(gen=20, variant=1)
        # algo = pg.pso_gen(gen=20)
        # algo = pg.bee_colony(gen=20)

        algo = pg.algorithm(algo)

        if not i:
            algo.set_verbosity(1)
        archi.push_back(algo=algo, pop=pop)

    archi.evolve(1)
    archi.wait_check()

    fs = archi.get_champions_f()
    xs = archi.get_champions_x()
    champion = xs[0]
    current_min = fs[0][0]
    for x, f in zip(xs, fs):
        if f < current_min:
            current_min = f[0]
            champion = x

    print(current_min)
    print(champion)

    prob.params = champion

    ref = tiff.imread(args.input_file1)
    trans = tiff.imread(args.input_file2)

    x = np.linspace(1, ref.shape[-1], ref.shape[-1])
    y = np.linspace(1, ref.shape[-2], ref.shape[-2])
    z = np.linspace(1, ref.shape[0], ref.shape[0])
    new_z = np.linspace(1, ref.shape[0], 2.5 * ref.shape[0])
    ext_z = int(50-1.25 * ref.shape[0])
    grid = np.array(np.meshgrid(new_z, y, x, indexing='ij'))
    grid = np.moveaxis(grid, 0, -1)

    ref_out = np.zeros((prob.a1.shape[0], ref.shape[1], ref.shape[2]))
    trans_out = np.zeros(ref_out.shape)

    ref_interp = interpn((z, y, x), ref[:, :, :], grid, 'linear')
    trans_interp = interpn((z, y, x), trans[:, :, :], grid, 'linear')
    ref_out[:, :, :] = np.pad(ref_interp, ((ext_z, ext_z), (0, 0), (0, 0)), mode='constant')
    trans_out[:, :, :] = scipy.ndimage.affine_transform(
        trans_interp, prob.affine_matrix, output_shape=(prob.a2.shape[0] + 2 * prob.ext_z,) + prob.a2.shape[1:])
    name = os.path.basename(args.output_file).split(".")[-2]
    np.savetxt(name + "_affine_matrix.txt", prob.affine_matrix)
    np.savetxt(name + "_champion.txt", prob.params)

    #TZCYXS order
    tiff.imwrite(args.output_file, trans_out[None, :, None, :, :].astype(np.float32), imagej=True)
    nome_org = os.path.basename(args.input_file1).split(".")[-2]+"_REF.tiff"
    tiff.imwrite(nome_org, ref_out[None, :, None, :, :].astype(np.float32), imagej=True)


if __name__ == '__main__':
    main()
