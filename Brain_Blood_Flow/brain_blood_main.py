import numpy as np
import matplotlib.pyplot as plt
from Physiology import Physiology


fn = 'D:/Biological Python Data/APC_012722-text.txt'
subj_1 = Physiology(fn)
subj_1.plot_co2(0, 20000)
subj_1.plot_mca(0, 20000)
subj_1.plot_bp(0, 20000)
subj_1.plot_ecg(0, 20000)