import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import sys

"""
This script is used to compute the accuracy of the speed estimation based on computer vision
"""


def main(argv):

    # =========================================
    # load the vehicle detection result based on computer vision
    wheel_data = np.load('wheel_dynamics.npy')
    # The labeled data starts from the fourth veh (7th wheel)

    wheel_times = wheel_data[6:, :, 0]     # relative timestamps
    wheel_h_loc = wheel_data[6:, :, 1]
    wheel_v_loc = wheel_data[6:, :, 2]
    veh_times = detect_vehicle_from_wheels(wheel_times)

    # =========================================
    # load the log data
    t_format = '%Y-%m-%d %H:%M:%S.%f'
    log_times = []
    log_speed = []  # mph
    with open('log.csv', 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split(',')
            log_times.append(datetime.strptime(items[0], t_format))
            log_speed.append(float(items[1]))

    # =========================================
    # load laser range finder data
    laser_times = []
    laser_dist = []
    with open('range_finder.txt','r') as f:
        for line in f:
            items = line.strip().split('#')
            laser_times.append(datetime.strptime(items[0], t_format))
            laser_dist.append(float(items[1]))

    laser_dist = clean_laser_data(laser_dist)
    # fixed the time offset
    t_offset = log_times[0] - laser_times[0] - timedelta(seconds=45)
    laser_times = [i + t_offset for i in laser_times]

    det_times, veh_dist = get_veh_dist(laser_times, laser_dist)

    # the labeled data starts from the 4th car
    del det_times[:3]
    del veh_dist[:3]

    # =========================================
    # visualize the range finder data, the log data and the video data
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    # plot the wheels
    for i in range(0, wheel_times.shape[0]):
        ax[0].plot(wheel_times[i, :], wheel_h_loc[i, :])

    # plot the vehicles
    for i in range(0, veh_times.shape[0]):
        ax[0].plot(veh_times[i, :], wheel_h_loc[i, :], linestyle='--')

    # plot the range finder data up to the
    ax[1].plot(laser_times, laser_dist)
    ax[1].set_ylabel('Laser distance (m)')

    for i, interval in enumerate(det_times):
        ax[1].plot(interval, [veh_dist[i], veh_dist[i]], '--', linewidth=4)

    # plot the speed
    dt = timedelta(seconds=1)
    ax2 = ax[1].twinx()
    for i, t in enumerate(log_times):
        ax2.plot([t-dt, t+dt], [log_speed[i], log_speed[i]], linewidth=2)

    ax2.set_ylabel('Speed (mph)')
    ax2.set_ylim([10,35])

    # =========================================
    # compute the estimation error
    est_speed = estimate_speed(veh_times, veh_dist)

    compute_accuracy(est_speed, log_speed, log_times)


    plt.show()


def detect_vehicle_from_wheels(wheel_times):
    """
    This function returns the vehicle detection from the wheels. Based on the analysis of the data, this function
    detects the vehicle by the following assumption:
        - If the average separation time for two consecutive wheels is less than 0.5s, then they belong to the same
        vehicle. Then return the average time of the those wheels as the passing time of the vehicle.
    :param wheel_times: the detected wheel times passing each pixel (in total 640 pixels).
    :return: np.array, num_veh x 640
    """
    veh_times = []

    candidate_veh = []
    for i, wheel in enumerate(wheel_times):
        if len(candidate_veh) == 0:
            candidate_veh.append(wheel)
            # print('added wheel {0}'.format(i))
            continue
        else:
            avg_t = np.average(wheel - np.array(candidate_veh[-1]))
            # print('avg_t between {0} and {1} wheels: {2} s'.format(i-1, i, avg_t))
            # compare with the current wheel and see if they belong to the same vehicle
            if avg_t <= 1.0:
                candidate_veh.append(wheel)
                # print('added wheel {0}'.format(i))
                continue
            else:
                # identified all wheels that belong to one vehicle
                # compute the average times and add to veh_times
                avg_times = np.average( np.array(candidate_veh), 0)
                veh_times.append(avg_times)
                # print('detected one vehicle')
                # clear and add the current wheel to the candidate vehicle
                del candidate_veh[:]
                candidate_veh.append(wheel)

    # make sure the last vehicle is added
    if len(candidate_veh) != 0:
        avg_times = np.average( np.array(candidate_veh), 0)
        veh_times.append(avg_times)

    return np.array(veh_times)


def clean_laser_data(laser_dist):
    """
    This function cleans the laser range finder data by:
        - removing all distances higher than 6 meters and lower then 1 meter.
    :param laser_dist: the raw distance list
    :return: np.array
    """
    laser_dist = np.array(laser_dist)/100.0

    laser_dist[laser_dist < 1.0] = 10
    laser_dist[laser_dist > 7.0] = 10

    return laser_dist


def get_veh_dist(laser_times, laser_dist):
    """
    This function returns the average distance to each vehicle
        - Consecutive points in the range of 2~7 meters will be considered as one vehicle
    :param laser_times: the times of the laser measures
    :param laser_dist: array of distance in meters
    :return:
    """
    veh_dist = []
    veh_intervals = []

    candidate_dist = []
    t_start = None
    t_end = None
    for i, dist in enumerate(laser_dist):

        if 2 <= dist <= 7:
            candidate_dist.append(dist)
            if t_start is None:
                t_start = laser_times[i]
            continue
        else:
            if t_start is not None:
                # found all points for one vehicle,
                # compute the median as the distance and return the time interval
                t_end = laser_times[i]

                if len(veh_intervals) !=0 and t_start-veh_intervals[-1][1]<timedelta(seconds=0.5):
                    # should be the same vehicle as the previous one, simply update the end time
                    veh_intervals[-1] = (veh_intervals[-1][0], t_end)
                else:
                    veh_intervals.append((t_start, t_end))
                    veh_dist.append(np.median(candidate_dist))

                # clear candidate points
                del candidate_dist[:]
                t_start = None

    # make sure get the last detection
    if len(candidate_dist) != 0:
        t_end = laser_times[-1]
        veh_intervals.append((t_start, t_end))
        veh_dist.append(np.median(candidate_dist))

    return veh_intervals, veh_dist


def estimate_speed(veh_times, veh_dist):
    """
    This function estimates the vehicle speed.
    NOTE:
        The input data must be aligned.
    :param veh_times: a list of veh_time, each veh_time is a 640 array given the relative time for passing each pixel
    :param veh_dist: a list of floats in meters
    :return: a list of estimated speed (mph)
    """

    est_speed = []

    coe = 1.0

    # assume speeds are constant
    for i in range(0, veh_times.shape[0]):
        v_pps = 640.0/(veh_times[i, -1] - veh_times[i, 0])  # pixels per second
        # est_speed.append( v_pps*coe*veh_dist[i] )           # m/s multiplied by coe
        est_speed.append( v_pps*coe )           # m/s multiplied by coe

    return np.array(est_speed)


def compute_accuracy(est_speed, log_speed, log_times):
    """
    This function computes the error for the estimated speed
    :param est_speed: np array, estimated speed
    :param log_speed: a list of floats, the logged true speed (mph)
    :param log_times: a list of datetime for the vehicles
    :return: RMSE and a plot
    """

    # make sure they are same length
    log_speed = np.array(log_speed)[0:len(est_speed)]
    log_times = np.array(log_times)[0:len(est_speed)]

    # compute the coefficient
    coe = np.mean(log_speed/est_speed)
    est_speed_mph = est_speed*coe

    # compute the rooted mean square error
    err = np.sqrt( np.sum((log_speed-est_speed_mph)**2)/len(log_speed) )
    print('Coefficient is {0}'.format(coe))
    print('RMSE is {0} mph'.format(err))

    # plot them
    fig = plt.figure(figsize=[15, 8], dpi=100)
    ax = fig.add_axes([0.1, 0.15, 0.82, 0.75])

    # plt.plot(log_times, est_speed_mph, color='r', linewidth=2, label='estimated speed')
    # plt.plot(log_times, log_speed, color='b', linewidth=2, label='true speed')

    plt.plot(est_speed_mph, color='r', linewidth=2, label='estimated speed')
    plt.plot(log_speed, color='b', linewidth=2, label='true speed')

    plt.legend()
    plt.xlabel('vehicles')
    plt.ylabel('speed (mph)')
    plt.title('RMSE: {0} mph'.format(err))

    plt.draw()



if __name__ == "__main__":
    sys.exit(main(sys.argv))