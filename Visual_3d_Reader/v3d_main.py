import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Biomechanics import Biomechanics

fn = 'D:/Alexis_Subject_16/S16 DH 12.txt'
s_16_dh_12 = Biomechanics(fn)
s_16_dh_12.plot_fz()