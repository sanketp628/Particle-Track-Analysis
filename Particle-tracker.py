import math
from math import sqrt, sin, cos, tan, degrees, radians, acos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


# IMPORT OF DATASET
dt = 0.1  # FRAME RATE
pixel12nm = 0.117  # PIXEL CONVERSION FACTOR (um)
df = pd.read_csv('1.csv')
frames_val = df.iloc[:, 0].values
x_val = df.iloc[:, 2].values
y_val = df.iloc[:, 3].values
frames = [i for i in range(0, len(frames_val))]
frames = np.array(frames)
x = np.empty(len(frames_val), dtype=float)
y = np.empty(len(frames_val), dtype=float)
for i in range(0, len(frames)):
    x[i] = x_val[i] * pixel12nm
    y[i] = y_val[i] * pixel12nm


# DIRECT MOTION TRAJECTORY ANALYSIS
# Number of trajectory points grouped together
points_grouped = 4
# Threshold ratio to distinguish between direct(>difTH) or non-direct (<difTH)
threshold_ratio = 0.7
# Minimum number of points to consider a pause [3 points for 0.1s fr rate -> 0.2s pausetime]
min_points = 5

# Displacement between adjacent frames (um)
disp = []
for i in range(0, len(frames_val) - 1):
    d = sqrt((x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2)
    disp.append(d)
disp = np.array(disp)
# print((disp))


# Displacement every "points_grouped" frames (um)
grouped_disp = []
for i in range(0, (len(frames_val) - points_grouped + 1)):
    d = sqrt((x[i + (points_grouped - 1)] - x[i])**2 +
             (y[i + (points_grouped - 1)] - y[i])**2)
    grouped_disp.append(d)
grouped_disp = np.array(grouped_disp)
# print(grouped_disp)


# Sum of displacements for first "points_grouped" points (um)
sum_disp = []
for i in range(0, len(disp) - (points_grouped - 2)):
    sum_disp.append(sum(disp[i:(i + (points_grouped - 1))]))
sum_disp = np.array(sum_disp)
# print((sum_disp))
ratio = np.true_divide(grouped_disp, sum_disp)
# print(ratio.size)

# Assign an average ratio to each data point
p = []
for i in range(0, len(ratio) + (points_grouped - 1)):
    if i == 0:
        pc = np.mean(ratio[0])
    elif i < points_grouped and i > 0:
        pc = np.mean(ratio[:i + 1])
    elif i > len(ratio):
        pc = np.mean(ratio[i - (points_grouped - 1):])
    else:
        pc = np.mean(ratio[i - (points_grouped - 1):i + 1])
    p.append(pc)
p = np.array(p)
# print(p)


# First screening to identify truly direct regions based on displacement ratio condition
direct_1 = []
for i in range(0, len(p)):
    if p[i] >= threshold_ratio:
        direct_1.append(1)
    else:
        direct_1.append(0)
direct_1 = np.array(direct_1)
# print(direct_1.shape)


# 2. Direct motion identification based on angle between adjacent tracsvectors

# Threshold angle to distinguish between direct(<threshold_angle) or non-direct (>threshold_angle)
threshold_angle = 90

disp_vector = []
for i in range(0, len(frames) - 1):
    d = ((x[i + 1] - x[i]), (y[i + 1] - y[i]))
    disp_vector.append(d)
disp_vector = np.array(disp_vector)

nv = []
for i in range(0, len(disp_vector)):
    d = np.linalg.norm(disp_vector[i, :])
    nv.append(d)
nv = np.array(nv)
# print(nv)

cos_theta = []
sign = []
for i in range(0, len(disp_vector) - 1):
    dot_mul = np.sum(disp_vector[i, :].conj(
    ) * disp_vector[i + 1, :], axis=0) / (nv[i] * nv[i + 1])
    cos_theta.append(dot_mul)

    mul = (disp_vector[i, 0] * disp_vector[i, 1])
    sign.append(np.sign(mul))

cos_theta = np.array(cos_theta)
sign = np.array(sign)
# print(cos_theta, sign)

theta_in_degrees = []
theta_in_degrees_sign = []
for i in range(0, len(cos_theta)):
    theta = acos(cos_theta[i]) * 180 / math.pi
    theta_in_degrees.append(theta)

    theta_in_degrees_sign.append(
        sign[i] * (acos(cos_theta[i]) * 180 / math.pi))

# Angle in degrees (without sign - mod value)
theta_in_degrees = np.array(theta_in_degrees)
# Angle in degrees (with sigh - Actual Angle)
theta_in_degrees_sign = np.array(theta_in_degrees_sign)
# print(theta_in_degrees_sign)

# Second screening to identify truly direct regions based on angle condition
direct_2 = []
direct_2.append(direct_1[0])

for i in range(0, len(theta_in_degrees)):
    if theta_in_degrees[i] < threshold_angle:
        direct_2.append(1)
    else:
        direct_2.append(0)

direct_2.append(direct_1[len(theta_in_degrees) + 1])
direct_2 = np.array(direct_2)
# print(direct_2)

# 3. Combine both screening methods (Angle & Displacement)
Angle_condition = True
direct_trajectory = []
if Angle_condition:
    for i in range(0, len(direct_2)):
        if direct_1[i] == 1 and direct_2[i] == 1:
            direct_trajectory.append(1)
        else:
            direct_trajectory.append(0)
else:
    direct_trajectory = direct_1

direct_trajectory = np.array(direct_trajectory)
# print(direct_trajectory)

# 4. Filter out short nondirect segments <min_points number of points (sequences of less than 5 zeros)
i = 0
idx = []

while i < len(direct_trajectory):
    if direct_trajectory[i] == 0:
        idx.append(i)
        if len(direct_trajectory) == i and not (not idx):
            if len(idx) < min_points:
                direct_trajectory[idx] = 1

        i = i + 1
    else:
        if not (not idx):
            if len(idx) + 1 < min_points:
                direct_trajectory[idx] = 1

            idx = []
            i = i + 1
        else:
            i = i + 1
# print((direct_trajectory[-1]))

# 5. Filter out direct segments < 0.5pixel max confinement radius
i = 0
s = 0
idx = []
D = []
while i < len(direct_trajectory):
    if direct_trajectory[i] == 1:
        idx.append(i)
        i = i + 1
        if len(direct_trajectory) - 1 == i and not (not idx):
            if idx[0] != 1:
                idx.extend((idx[0] - 1, idx[-1] + 1))
            else:
                idx.append(idx[-1] + 1)
            for s in range(1, len(idx)):
                D.append(sqrt((x[idx[s]] - x[idx[0]])**2 +
                              (y[idx[s]] - y[idx[0]])**2))
            R_max = max(D)

            if (len(idx) < 12 and R_max <= 0.5 * pixel12nm):
                direct_trajectory[idx] = 0
    else:
        if not (not idx):
            if idx[0] != 1:
                idx.extend((idx[0] - 1, idx[-1] + 1))
            else:
                idx.append(idx[-1] + 1)
            for s in range(0, len(idx)):
                D.append(sqrt((x[idx[s]] - x[idx[0]])**2 +
                              (y[idx[s]] - y[idx[0]])**2))
            R_max = max(D)

            if (len(idx) < 12 and R_max <= 0.5 * pixel12nm):
                direct_trajectory[idx] = 0
            idx = []
            D.clear()
            R_max = None
            i = i + 1
        else:
            i = i + 1

# print(direct_trajectory)


# 6. Find non-direct indices (diffusive or confined motion)

non_direct_idx = np.where(direct_trajectory == 0)
# +1 For Python Syntax correction as Python numbering starts from 0
non_direct_idx = np.array(non_direct_idx[0] + 1)

non_direct_start = []
non_direct_end = []


if not(not non_direct_idx.all):
    non_direct_start.append(non_direct_idx[0])
    # j = 0
    # k = 1
# range starting from 1 as range does not include the last number in Python
# so taking values from range(1, len(non_direct_idx)) rather than range(0, len(non_direct_idx)-1)
    for i in range(1, len(non_direct_idx)):
        if i != len(non_direct_idx) - 1:
            if non_direct_idx[i] - non_direct_idx[i - 1] > 1:
                non_direct_end.append(non_direct_idx[i - 1])
                # j = j + 1
                non_direct_start.append(non_direct_idx[i])
                # k = k + 1

        else:
            if non_direct_idx[i] - non_direct_idx[i - 1] > 1:
                non_direct_end.append(non_direct_idx[i - 1])
            else:
                non_direct_end.append(non_direct_idx[i])

else:
    non_direct_start = []
    non_direct_end = []

non_direct_start = np.array(non_direct_start)
non_direct_end = np.array(non_direct_end)
# print(non_direct_idx)
# print(type(non_direct_start))
# print(non_direct_end)

# 7. Indices for direct parts
# direct_start = []
# direct_end=[]


if not non_direct_idx.all:
    direct_start = 0
    direct_end = len(x)
else:
    direct_start = []
    direct_end = []

    for i in range(0, len(non_direct_start)):
        if i == 0:
            if non_direct_start[i] != 1:
                direct_start.append(1)
                direct_end.append(non_direct_start[i])
        else:
            direct_start.append(non_direct_end[i - 1])
            direct_end.append(non_direct_start[i])

    if non_direct_end[len(non_direct_end) - 1] < len(x) - 3:
        direct_start.append(non_direct_end[len(non_direct_end) - 1])
        direct_end.append(len(x))

direct_start = np.array(direct_start)
direct_end = np.array(direct_end)
# print(direct_start)
# print(direct_end)

# CALCULATIONS

# Time difference between frames(Delta Time)
delta_time = []
for i in range(0, len(frames) - 1):
    delta_time.append(np.multiply((frames[i + 1] - frames[i]), dt))


# Instantaneous Speed (um/s)
velocity = np.true_divide(disp, delta_time)

Rc = []
Rc_max = []

track_info = [None] * 5
track_info[0] = ['f']
DATASUBHEADER_keys = ['Phase type', 'idx', 'xy(pixels)', 'Instantaneous Runlength(um)',
                      'Total Runlength(um)', 'Instantaneous speed(um/s)',
                      'Average speed(um/s)', 'Duration(s)', 'D (um2/s)', 'Alpha',
                      'Displacement start-end(nm)', 'Maximum confinement radius(nm)']
DATASUBHEADER = dict.fromkeys(DATASUBHEADER_keys, None)
track_info[1] = DATASUBHEADER
# print(track_info[1]['idx'])
s = 0
m = 1
idx_start = []
idx_end = []
# print(non_direct_start, direct_start)

if not(not non_direct_start.all()) and not (not direct_start.all()):
    idx_start = np.sort(np.concatenate((non_direct_start, direct_start)))
    idx_end = np.sort(np.concatenate((non_direct_end, direct_end)))
elif not non_direct_start:
    idx_start = direct_start
    idx_end = direct_end
else:
    idx_start = non_direct_start
    idx_end = non_direct_end

idx_start = np.append(idx_start, idx_start[0])
idx_end = np.append(idx_end, idx_end[-1])
length_idx = idx_end - idx_start + 1
# print(idx_start)
# arr_1 = []
# for a in range(idx_start[1], idx_end[1] + 1):
#
#     arr_1.append(a)
# print(arr_1)

track_info[1]['Alpha'] = []
for i in range(0, len(idx_start)):
    # PARAMETER TRANSPORT

    # Indices to distinguish direct vs diffusive or Frame ('idx')
    arr_1 = []
    # farr_1 = []
    for a in range(idx_start[i], idx_end[i] + 1):
        arr_1 = np.append(arr_1, a)
    #     farr_1.append(arr_1)
    track_info[1]['idx'] = arr_1
    # print(track_info[1]['idx'])

    # xy coordinates ('xy(pixels)')
    arr_2_x = []
    arr_2_y = []
    arr_2 = []
    for b in range(idx_start[i], idx_end[i] + 1):
        arr_2_x = np.append(arr_2_x, x[b - 1])
        arr_2_y = np.append(arr_2_y, y[b - 1])
        arr_2 = np.append(arr_2_x, arr_2_y)
    track_info[1]['xy(pixels)'] = arr_2
    # print(track_info[1]['xy(pixels)'])

    # Instantaneous runlength (um) ('Instantaneous Runlength(um)')
    arr_3 = []
    for c in range(idx_start[i], idx_end[i]):
        arr_3 = np.append(arr_3, disp[c - 1])
    track_info[1]['Instantaneous Runlength(um)'] = arr_3
    # print(track_info[1]['Instantaneous Runlength(um)'])

    # Total runlength per segment ('Total Runlength(um)')
    track_info[1]['Total Runlength(um)'] = sum(
        disp[idx_start[i] - 1:idx_end[i]])
    # print(track_info[1]['Total Runlength(um)'])

    # Instantaneous speed ('Instantaneous speed(um/s)')
    track_info[1]['Instantaneous speed(um/s)'] = velocity[idx_start[i] -
                                                          1:idx_end[i] - 1]
    # print(track_info[1]['Instantaneous speed(um/s)'])

    # Average speed per segment ('Average speed(um/s)')
    track_info[1]['Average speed(um/s)'] = np.mean(
        velocity[idx_start[i] - 1:idx_end[i] - 1])
    # print(track_info[1]['Average speed(um/s)'])

    # Time spent (PAUSING for diffusive & PROCESSIVITY for direct) ('Duration(s)')
    track_info[1]['Duration(s)'] = dt * ((length_idx[i]) - 1)
    # print(track_info[1]['Duration(s)'])

    x_c = x[idx_start[i] - 1:idx_end[i]]
    y_c = y[idx_start[i] - 1:idx_end[i]]
    D = np.empty(len(x_c))
    for s in range(0, len(x_c)):
        D[s] = (sqrt((x_c[s] - x_c[0])**2 +
                     (y_c[s] - y_c[0])**2))

    R_max = max(D) * 1000
    Rc_max.append(R_max)  # Distance between most separated points
    # print(Rc_max)

    D_2 = sqrt((x[idx_end[i] - 1] - x[idx_start[i] - 1])**2 +  # (-1) in both to compensate for the '0' starting point of Python
               (y[idx_end[i] - 1] - y[idx_start[i] - 1])**2)
    Rc.append(D_2 * 1000)  # Distance between start and end points
    # print(Rc)

    # Confinement diameter (nm) start-end ('Displacement start-end(nm)')
    track_info[1]['Displacement start-end(nm)'] = np.array(Rc)

    # Confinement diameter (nm) maximum ('Maximum confinement radius(nm)')
    track_info[1]['Maximum confinement radius(nm)'] = np.array(Rc_max)

    # MSD ANALYSIS
    msd = []
    tau = []
    coef = []
    coef_pl = []
    if length_idx[i] >= 12:  # Minimum number of points to calculate MSD
        for j in range(1, length_idx[i]):
            tau.append(j * dt)
            # x_msd_start = x[idx_start[i] + j:idx_end[i] + 1]
            # y_msd_start = y[idx_start[i] + j:idx_end[i] + 1]
            # x_msd_end = x[idx_start[i]:idx_end[i] - j + 1]4
            # y_msd_end = y[idx_start[i]:idx_end[i] - j + 1]
            # for h in range(0, len(x_msd_start)):  # np.diff can also be used
            #     xdata = (x_msd_start[h] - x_msd_end[h])
            #     ydata = (y_msd_start[h] - y_msd_end[h])
            # msd.append(np.mean(xdata**2 + ydata**2))
            x = x.reshape((len(x), 1))
            y = y.reshape((len(y), 1))
            t1 = np.power(x[idx_start[i] + j: idx_end[i], :] -
                          x[idx_start[i]: idx_end[i] - j, :], 2)
            t2 = np.power(y[idx_start[i] + j: idx_end[i], :] -
                          y[idx_start[i]: idx_end[i] - j, :], 2)
            msd.append(np.mean(t1 + t2, axis=0)[0])

        tau = np.transpose(tau)
        msd = np.transpose(msd)
        thr = 5
        # assp = np.empty((5, 1), float)
        tau_thr = tau[0:thr].reshape(thr, 1)
        msd_thr = msd[0:thr].reshape(thr, 1)
        mtx = np.ones(len(tau[0:thr])).reshape(thr, 1)
        # for g in range(0, len(tau_thr)):
        #     mtx.append([tau_thr[g], 1], axis=1)
        assp = np.append(tau_thr, mtx, axis=1)
        log_assp = np.append(np.log10(tau_thr), mtx, axis=1)
        coef.append(np.linalg.lstsq(assp, msd_thr, rcond=None)[0])
        coef_pl.append(np.linalg.lstsq(
            log_assp, np.log10(msd_thr), rcond=None)[0][0])
        # print(coef)
        if coef_pl[0][0] > 0:  # Alpha coefficient > 0
            track_info[1]['D (um2/s)'] = coef[0] / 4
            track_info[1]['Alpha'].append(coef_pl[0][0])
        else:
            track_info[1]['D (um2/s)'] = None
            track_info[1]['Alpha'].append(np.nan)
    else:
        track_info[1]['D (um2/s)'] = None
        track_info[1]['Alpha'].append(np.nan)

    # print(round(np.nan * 100) / 100)
# print(idx_start[i])

# Assign Phase Type
    if i == len(idx_start):
        track_info[1]['Phase type'] = 'Full Track'
    elif direct_start.any() == idx_start[i]:
        if not np.isnan(track_info[1]['Alpha'][-1]):
            if round(track_info[1]['Alpha'][-1] * 100) / 100 >= 1.45:
                if track_info[1]['Average speed(um/s)'] > 0.25:
                    track_info[1]['Phase type'] = 'Direct-Fast'
                else:
                    track_info[1]['Phase type'] = 'Direct-Slow'

            elif (track_info[1]['Alpha'][-1] * 100) / 100 >= 1.0 and (track_info[1]['Alpha'][-1] * 100) / 100 < 1.45:
                track_info[1]['Phase type'] = 'Diffusive'
                if track_info[1]['Average speed(um/s)'] > 0.25:
                    track_info[1]['Phase type'] = 'Diffusive CHECK'
                else:
                    track_info[1]['Phase type'] = 'Diffusive'

        else:
            if track_info[1]['Average speed(um/s)'] > 0.25:
                track_info[1]['Phase type'] = 'Direct-Short-Fast'
            else:
                track_info[1]['Phase type'] = 'Direct-Short-Slow'

    else:
        if not np.isnan(track_info[1]['Alpha'][-1]):
            if (track_info[1]['Alpha'][-1] * 100) / 100 < 1:
                track_info[1]['Phase type'] = 'Confined'
            elif (track_info[1]['Alpha'][-1] * 100) / 100 >= 1.0 and (track_info[1]['Alpha'][-1] * 100) / 100 < 1.45:
                track_info[1]['Phase type'] = 'Diffusive'
            else:
                track_info[1]['Phase type'] = 'Recategorize to Dir'

        else:
            track_info[1]['Phase type'] = 'Confined-Short'

    if track_info[1]['Phase type'] == 'Direct-Fast' or track_info[1]['Phase type'] == 'Direct-Short-Fast':
        plt.plot(x[idx_start[i]: idx_end[i]],
                 y[idx_start[i]: idx_end[i]], '-*', color='blue', label='Direct Fast')
    elif track_info[1]['Phase type'] == 'Direct-Slow' or track_info[1]['Phase type'] == 'Direct-Short-Slow':
        plt.plot(x[idx_start[i]: idx_end[i]],
                 y[idx_start[i]: idx_end[i]], '-*', color='cyan', label='Direct Slow')
    elif track_info[1]['Phase type'] == 'Confined' or track_info[1]['Phase type'] == 'Confined-Short':
        plt.plot(x[idx_start[i]: idx_end[i]],
                 y[idx_start[i]: idx_end[i]], '-*', color='red', label='Confined')
    elif track_info[1]['Phase type'] == 'Diffusive':
        plt.plot(x[idx_start[i]: idx_end[i]],
                 y[idx_start[i]: idx_end[i]], '-*', color='magenta', label='Diffusive')

plt.xlabel('X - coordinates(um)')
plt.ylabel('Y - coordinates(um)')
plt.title('Particle Trajectory')
plt.legend()
# plt.colorbar(mappable=np.array(frames))
plt.grid()
plt.show()
