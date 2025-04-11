import numpy as np
import matplotlib.pyplot as plt


def save_stats_long(self, stat_file_path):
    fn = stat_file_path + 'Loran Stats LONG.csv'
    with open(fn, 'a') as stat_file:
        for rep in range(self.n_reps):
            stat_file.write(
                self.subject + ',' + self.cond + ',' + self.rom + ',' + self.trial + ',' + str(rep) + ',' + str(
                    self.peak_torque[rep]) + ',' + str(self.stiffness[rep]) + ',' + str(
                    self.energy_absorbed[rep]) + ',' + str(self.energy_returned[rep]) + '\n')
    stat_file.close()

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

def bin_counter(rounded_dbp, MSNA, nerve_on, bin_width=1):
    max_bp = rounded_dbp[len(rounded_dbp) - 1]
    min_bp = rounded_dbp[0]
    beat_range = int(max_bp - min_bp) + 1
    n_beats = np.zeros(beat_range, dtype=float)  # need to divide by bin_width
    bp = np.zeros(beat_range, dtype=float)
    current_bin = min_bp
    bp_cntr = 0
    nerve_cntr = 0
    k = 0
    for i in range(len(rounded_dbp)):
        if rounded_dbp[i] == current_bin:
            bp_cntr += 1
            if nerve_on[i]:
                nerve_cntr += 1
        else:
            bp[k] = rounded_dbp[i-1]
            #print(k, bp_cntr)
            if bp_cntr != 0:
                n_beats[k] = 100.* nerve_cntr / bp_cntr
            else:
                n_beats[k] = 0
            k += 1
            current_bin += bin_width
            bp_cntr = 0
            nerve_cntr = 0
    return n_beats, bp
file_name = 'D:/Biological Python Data/01P56BW_sBRs.csv'  #will put file here when i convert the data to tab delimetied files

alligned_data = np.genfromtxt(file_name, delimiter=',', skip_header=1)

dbp = alligned_data[:, 0]
MSNA = alligned_data[:, 1]
time = alligned_data[:, 2]

rounded_dbp = np.round(dbp)

nerve_on_zeros = np.zeros(len(dbp), dtype=bool)
nerve_on = nerve_on_zeros.astype(bool)
#print(MSNA)


for i in range(len(MSNA)):
    if MSNA[i] > 0:
        nerve_on[i] = True  #assign true for MSNA for that cardiac cycle indicating a burst
    else:
        nerve_on[i] = False #labels no burst as false for that cardiac cycle
#print('before_sort')
#for i in range(10):
#    print(i, dbp[i], MSNA[i], time[i], nerve_on[i])

ind = np.argsort(rounded_dbp)
rounded_dbp = rounded_dbp[ind]
MSNA = MSNA[ind]
nerve_on = nerve_on[ind]
#rint('after_sort')

bin_counter(rounded_dbp, MSNA, nerve_on, bin_width=1)

"""
for i in range(25):
    print(i, rounded_dbp[i], MSNA[i], nerve_on[i])

plt.scatter(rounded_dbp, MSNA, color='r')
plt.xlabel('Diastolic')
plt.ylabel('MSNA')
plt.show()
"""