import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Biomechanics import Biomechanics

fn = 'D:/Alexis_Subject_27/S27 Neutral 08.txt'
s_27_n_08 = Biomechanics(fn)
s_27_n_08.get_stance()

s_27_n_08.plot_fz()