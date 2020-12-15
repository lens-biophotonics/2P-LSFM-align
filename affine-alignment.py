import argparse

import numpy as np
import pygmo as pg
import scipy.ndimage
from scipy.interpolate import interpn
import tifffile as tiff


class AlignmentProblem:
    def __init__(self, input_file1, input_file2):
        self.T = np.zeros(4)
        self.T[-1] = 1
        self.angles = np.zeros(3)
        self.zoom = 1
        self.input_file1 = input_file1
        self.input_file2 = input_file2
        self.output_file = None
        self.a2_min = None
        self.centerT = None

        self._load_files()

    def _load_files(self):
        self.a1 = tiff.imread(self.input_file1)[:, 0]
        self.a2 = tiff.imread(self.input_file2)[:, 0]

        self._interpolate()

        self.a1 = self.a1 > 0
        self.a2_min = np.min(self.a2[self.a2 > 0])
        self.centerT = np.r_[np.array(self.a2.shape) / 2, 1]

    def _interpolate(self):
        x = np.linspace(1, self.a1.shape[2], self.a1.shape[2])
        y = np.linspace(1, self.a1.shape[1], self.a1.shape[1])
        z = np.linspace(1, self.a1.shape[0], self.a1.shape[0])
        new_z = np.linspace(1, self.a1.shape[0], 2.5 * self.a1.shape[0])

        grid = np.array(np.meshgrid(new_z, y, x, indexing='ij'))
        grid = np.moveaxis(grid, 0, -1)

        self.a1 = interpn((z, y, x), self.a1, grid, 'nearest')
        self.a2 = interpn((z, y, x), self.a2, grid, 'nearest')

    @property
    def params(self):
        return np.r_[self.T[:3], self.angles, self.zoom]

    @params.setter
    def params(self, x):
        self.T[:3] = x[:3]
        self.angles = x[3:6]
        self.zoom = x[-1]

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

        M = np.dot(R, Z)
        M[:3, -1] += self.T[:3]

        return M

    @staticmethod
    def get_bounds():
        a = np.array([
            [-10, 10],     # Tz
            [-100, 100],   # Ty
            [-100, 100],   # Tx
            [-np.pi / 2, np.pi / 2],     # theta1
            [-np.pi / 2, np.pi / 2],     # theta2
            [-np.pi / 2, np.pi / 2],     # theta3
            [0.66, 2],     # zoom
        ])

        bounds = (a[:, 0], a[:, 1])
        return bounds

    def fitness(self, x):
        self.params = x
        a2 = scipy.ndimage.affine_transform(self.a2, self.affine_matrix)
        a2[a2 < self.a2_min] = 0

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
    args = parse_args()
    prob = AlignmentProblem(args.input_file1, args.input_file2)

    archi = pg.archipelago(n=0)

    pop = pg.population(prob=prob, size=5)
    for i in range(0, 8):
        # each algo gets a different seed
        algo = pg.simulated_annealing(Ts=10., Tf=1, n_T_adj=2)
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
    a = scipy.ndimage.affine_transform(prob.a2, prob.affine_matrix)
    a[a < prob.a2_min] = 0
    tiff.imwrite(args.output_file, a.astype(np.float32))


if __name__ == '__main__':
    main()
