import numpy as np
import matplotlib.pyplot as plt
from BiomechTools import low_pass, add_padding

def get_min(curve, first_pt, last_pt):
    min_location = first_pt
    min_val = curve[first_pt]
    for i in range(first_pt, last_pt):
        if curve[i] < min_val:
            min_location = i
            min_val = curve[i]
    return min_val

def get_max(curve, first_pt, last_pt):
    max_location = first_pt
    max_val = curve[first_pt]
    for i in range(first_pt, last_pt):
        if curve[i] > max_val:
            max_location = i
            max_val = curve[i]
    return max_val

def save_stats_long(stat_file_path, subject, cond, ml_range):
    fn = stat_file_path + 'Balance LONG.csv'
    with open(fn, 'a') as stat_file:
        stat_file.write(
            subject + ',' + cond + ',' + str(ml_range) + ',' + '\n')
    stat_file.close()


filename = 'd:/Accelerometer Data/S14C3T1.csv'
subject = '14'
#cond = 'eyes_open'
cond = 'eyes_closed'
data = np.genfromtxt(filename, delimiter=',', skip_header=2)
t = data[:,0]
ax = data[:,1]
ay = data[:,2]
az = data[:,3]
a = data[:,4]
sampling_rate = 500.0
sm_az = low_pass(az, sampling_rate, 3.2)
sm_ax = low_pass(ax, sampling_rate, 3.2)
sm_ay = low_pass(ay, sampling_rate, 3.2)
sm_a = low_pass(a, sampling_rate, 3.2)

max = get_max(sm_az, 100, len(sm_az)-100)
min = get_min(sm_az, 100, len(sm_ax)-100)
range = 100.0 * (max - min) # M/L accel in cm/s
save_stats_long('d:/Accelerometer Data/', subject, cond, range)
#print('range ', max-min)

#plt.plot(sm_az[160000:180000], color='r', linewidth=3.0, label='smooth')
#plt.plot(az[160000:180000], color='gray', linewidth=.8, label='raw')
#plt.grid(True)
#plt.legend()
#plt.show()