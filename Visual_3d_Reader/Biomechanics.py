import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import degrees, asin
from BiomechTools import low_pass, zero_crossing, max_min, simpson_nonuniform, critically_damped, residual_analysis

class Biomechanics:

    def __init__(self, filename):
        with open(filename) as f:
            f.readline()  # skip first line
            self.var_name = f.readline().rstrip().split()
            self.n_vars = len(set(self.var_name))  # use set to count # of unique values in a list
            self.var_name.insert(0, 'pt_num')
            f.readline()  # skip next line
            f.readline()  # skip next line
            xyz = f.readline().rstrip().split()
            for i in range(0, len(self.var_name)):
                self.var_name[i] = self.var_name[i] + '_' + xyz[i]
            self.var_name[0] = 'pt_num'

            data = np.genfromtxt(filename, delimiter='\t', skip_header=5)
        self.n_rows = data.shape[0]  # number of rows of array
        self.n_cols = data.shape[1]
        self.n_steps = int(
            (self.n_cols - 1) / 3 / self.n_vars)  # drop first col, divide by 3 (x,y,z) then divide by # variables in file
        self.v3d = pd.DataFrame(data, columns=self.var_name)

    def get_stance(self):
        self.v3d.Lf, self.v3d.Lf_rf = zero_crossing(self.v3d.FP1_Z, 16, 0, self.n_rows-1)
        self.v3d.Rt, self.v3d.Rt_rf = zero_crossing(self.v3d.FP2_Z, 16, 0, self.n_rows-1)

    def plot_first_step(self):
        plt.plot(self.v3d.FP1_Z, 'r', label='FP1 Z')
    def plot_fz(self):
        plt.plot(self.v3d.FP2_Z, 'r', label='FP2 Z')
        plt.plot(self.v3d.FP1_Z, 'b', label='FP1 Z')
        plt.grid(True)
        plt.legend()
        plt.show()
