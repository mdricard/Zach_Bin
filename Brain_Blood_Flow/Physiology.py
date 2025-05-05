import numpy as np
import matplotlib.pyplot as plt
from BiomechTools import low_pass, critically_damped


class Physiology:
    def __init__(self, filename):
        data = np.genfromtxt(filename, delimiter='\t', skip_header=9)
        self.ecg = data[:, 1]       # ecg
        self.bp = data[:, 2]        # arterial blood pressure (mmHg)
        self.co2 = data[:, 3]       # CO2 (mmHg)
        self.mca = data[:, 4]       # Brain blood flow (cm/s)
        self.sampling_rate = 1000   # sampling rate in Hz
        self.smooth_co2 = critically_damped(self.co2, self.sampling_rate, 1.8)
        #for i in range(10):
        #    print(self.ecg[i], self.bp[i], self.co2[i], self.tcd[i])

    def plot_co2_mca(self):
        plt.plot(self.mca, 'r-', label='mca')
        plt.plot(self.co2, 'b-', label='co2')
        plt.xlabel('Time (ms)')
        plt.ylabel('MCA, co2 (mmHg)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_mca(self, first_pt, last_pt):
        plt.plot(self.mca[first_pt:last_pt], 'r-')
        plt.xlabel('Time (ms)')
        plt.ylabel('MCA (cm/s)')
        plt.grid(True)
        plt.show()

    def plot_co2(self, first_pt, last_pt):
        plt.plot(self.co2[first_pt:last_pt], 'b-', label='co2')
        plt.plot(self.smooth_co2[first_pt:last_pt], 'r-', label='Smooth co2')
        plt.xlabel('Time (ms)')
        plt.ylabel('Co2 (mmHg)')
        plt.grid(True)
        plt.show()

    def plot_ecg(self, first_pt, last_pt):
        plt.plot(self.ecg[first_pt:last_pt], 'b-')
        plt.xlabel('Time (ms)')
        plt.ylabel('ECG (mv)')
        plt.grid(True)
        plt.show()

    def plot_bp(self, first_pt, last_pt):
        plt.plot(self.bp[first_pt:last_pt], 'b-')
        plt.xlabel('Time (ms)')
        plt.ylabel('BP (mmHg)')
        plt.grid(True)
        plt.show()