# -*- coding: utf-8 -*-
#
# ,---------,       ____  _ __
# |  ,-^-,  |      / __ )(_) /_______________ _____  ___
# | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
# | / ,--'  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
# Copyright (C) 2023 Bitcraze AB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, in version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
"""
Example of how to connect to a motion capture system and feed the position to a
Crazyflie, using the motioncapture library. The motioncapture library supports all major mocap systems and provides
a generalized API regardless of system type.
The script uses the high level commander to upload a trajectory to fly a figure 8.

Set the uri to the radio settings of the Crazyflie and modify the
mocap setting matching your system.
"""
import argparse
import time
from threading import Thread
from pathlib import Path
import os
from datetime import datetime
from FileLogger import FileLogger

import motioncapture

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.positioning.motion_commander import MotionCommander  
from cflib.positioning.position_hl_commander import PositionHlCommander
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper


# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EF')

# The host name or ip address of the mocap system
host_name = '192.168.209.81'

# The type of the mocap system
# Valid options are: 'vicon', 'optitrack', 'optitrack_closed_source', 'qualisys', 'nokov', 'vrpn', 'motionanalysis'
mocap_system_type = 'optitrack'

# The name of the rigid body that represents the Crazyflie
rigid_body_name = 'Chaoxiang'

# True: send position and orientation; False: send position only
send_full_pose = True

# When using full pose, the estimator can be sensitive to noise in the orientation data when yaw is close to +/- 90
# degrees. If this is a problem, increase orientation_std_dev a bit. The default value in the firmware is 4.5e-3.
orientation_std_dev = 8.0e-3

# The trajectory to fly
# See https://github.com/whoenig/uav_trajectories for a tool to generate
# trajectories

# Duration,x^0,x^1,x^2,x^3,x^4,x^5,x^6,x^7,y^0,y^1,y^2,y^3,y^4,y^5,y^6,y^7,z^0,z^1,z^2,z^3,z^4,z^5,z^6,z^7,yaw^0,yaw^1,yaw^2,yaw^3,yaw^4,yaw^5,yaw^6,yaw^7
# figure8 = [
#     [1.050000, 0.000000, -0.000000, 0.000000, -0.000000, 0.830443, -0.276140, -0.384219, 0.180493, -0.000000, 0.000000, -0.000000, 0.000000, -1.356107, 0.688430, 0.587426, -0.329106, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.710000, 0.396058, 0.918033, 0.128965, -0.773546, 0.339704, 0.034310, -0.026417, -0.030049, -0.445604, -0.684403, 0.888433, 1.493630, -1.361618, -0.139316, 0.158875, 0.095799, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.620000, 0.922409, 0.405715, -0.582968, -0.092188, -0.114670, 0.101046, 0.075834, -0.037926, -0.291165, 0.967514, 0.421451, -1.086348, 0.545211, 0.030109, -0.050046, -0.068177, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.700000, 0.923174, -0.431533, -0.682975, 0.177173, 0.319468, -0.043852, -0.111269, 0.023166, 0.289869, 0.724722, -0.512011, -0.209623, -0.218710, 0.108797, 0.128756, -0.055461, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.560000, 0.405364, -0.834716, 0.158939, 0.288175, -0.373738, -0.054995, 0.036090, 0.078627, 0.450742, -0.385534, -0.954089, 0.128288, 0.442620, 0.055630, -0.060142, -0.076163, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.560000, 0.001062, -0.646270, -0.012560, -0.324065, 0.125327, 0.119738, 0.034567, -0.063130, 0.001593, -1.031457, 0.015159, 0.820816, -0.152665, -0.130729, -0.045679, 0.080444, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.700000, -0.402804, -0.820508, -0.132914, 0.236278, 0.235164, -0.053551, -0.088687, 0.031253, -0.449354, -0.411507, 0.902946, 0.185335, -0.239125, -0.041696, 0.016857, 0.016709, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.620000, -0.921641, -0.464596, 0.661875, 0.286582, -0.228921, -0.051987, 0.004669, 0.038463, -0.292459, 0.777682, 0.565788, -0.432472, -0.060568, -0.082048, -0.009439, 0.041158, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [0.710000, -0.923935, 0.447832, 0.627381, -0.259808, -0.042325, -0.032258, 0.001420, 0.005294, 0.288570, 0.873350, -0.515586, -0.730207, -0.026023, 0.288755, 0.215678, -0.148061, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
#     [1.053185, -0.398611, 0.850510, -0.144007, -0.485368, -0.079781, 0.176330, 0.234482, -0.153567, 0.447039, -0.532729, -0.855023, 0.878509, 0.775168, -0.391051, -0.713519, 0.391628, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],  # noqa
# ]

#obtain position and rotation from mocap
class MocapWrapper(Thread):
    def __init__(self, body_name):
        Thread.__init__(self)

        self.body_name = body_name
        self.on_pose = None
        self._stay_open = True

        self.start()

    def close(self):
        self._stay_open = False

    def run(self):
        mc = motioncapture.connect(mocap_system_type, {'hostname': host_name})
        while self._stay_open:
            mc.waitForNextFrame()
            for name, obj in mc.rigidBodies.items():
                if name == self.body_name:
                    if self.on_pose:
                        pos = obj.position
                        #call back on pose
                        self.on_pose([pos[0], pos[1], pos[2], obj.rotation])


def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def send_extpose_quat(cf, x, y, z, quat):
    """
    Send the current Crazyflie X, Y, Z position and attitude as a quaternion.
    This is going to be forwarded to the Crazyflie's position estimator.
    Note the transformation from Optitrack y-up to Crazyflie's ENU
    """
    if send_full_pose:
        cf.extpos.send_extpose(z, x, y, quat.z, quat.x, quat.y, quat.w)
    else:
        cf.extpos.send_extpos(z, x, y)


def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    # time.sleep(1)
    wait_for_position_estimator(cf)


def adjust_orientation_sensitivity(cf):
    cf.param.set_value('locSrv.extQuatStdDev', orientation_std_dev)


def activate_kalman_estimator(cf):
    cf.param.set_value('stabilizer.estimator', '2')

    # Set the std deviation for the quaternion data pushed into the
    # kalman filter. The default value seems to be a bit too low.
    cf.param.set_value('locSrv.extQuatStdDev', 0.06)


def activate_mellinger_controller(cf):
    cf.param.set_value('stabilizer.controller', '2')


def upload_trajectory(cf, trajectory_id, trajectory):
    trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
    trajectory_mem.trajectory = []

    total_duration = 0
    for row in trajectory:
        duration = row[0]
        x = Poly4D.Poly(row[1:9])
        y = Poly4D.Poly(row[9:17])
        z = Poly4D.Poly(row[17:25])
        yaw = Poly4D.Poly(row[25:33])
        trajectory_mem.trajectory.append(Poly4D(duration, x, y, z, yaw))
        total_duration += duration

    trajectory_mem.write_data_sync()
    cf.high_level_commander.define_trajectory(trajectory_id, 0, len(trajectory_mem.trajectory))
    return total_duration


def run_sequence(cf, trajectory_id, duration):
    commander = cf.high_level_commander

    commander.takeoff(1.0, 2.0)
    time.sleep(3.0)
    relative = True
    commander.start_trajectory(trajectory_id, 1.0, relative)
    time.sleep(duration)
    commander.land(0.0, 2.0)
    time.sleep(2)
    commander.stop()

def move_linear_MC(scf,velocity):

    with MotionCommander(scf,default_height=0.5) as commander:

        commander.forward(0.5, velocity)
        time.sleep(2)

        commander.right(0.5,velocity)
        time.sleep(2)

        commander.back(0.5, velocity)
        time.sleep(2)

        commander.land(0.2)

        commander.stop()

def position_hl_control(scf,Velocity):
    with PositionHlCommander(
            scf,
            x=0.0, y=0.0, z=0.0,
            default_velocity=Velocity,
            default_height=0.5,
            controller=PositionHlCommander.CONTROLLER_PID) as pc:
        # Go to a coordinate
        time.sleep(2)
            
        # move forward 
        pc.forward(0.5)

        time.sleep(2)

        # move right
        pc.right(0.5)

        time.sleep(2)

        # move backward
            
        pc.back(0.5)

        time.sleep(4)

        pc.land()


def take_off_simple(scf):
    with MotionCommander(scf, default_height=0.5) as mc:
        time.sleep(3)
        mc.stop()



def move_linear_HC(scf,Flight_time):

    flight_time = Flight_time

    commander = scf.cf.high_level_commander

    commander.takeoff(1.05, 5)
    time.sleep(5)

    commander.go_to(0.5, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time+1)

    commander.go_to(0, -0.5, 0, 0, flight_time, relative=True)
    time.sleep(flight_time+1)

    commander.go_to(-0.5, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time+1)

    commander.land(0, 5)
    time.sleep(5)

    commander.stop()

def get_filename():
    fileroot = args["fileroot"] 
        
    if args["filename"] is not None:
        name = args["filename"] + ".csv"
        fname = os.path.normpath(os.path.join(
            os.getcwd(), fileroot, name))
        i = 0
        while os.path.isfile(fname):
            i = i + 1
            name = args["filename"] + "_" + str(i) + ".csv"
            fname = os.path.normpath(os.path.join(
                os.getcwd(), fileroot, name))

    else:
        # get relevant arguments
        # keywords = args["keywords"]
        # estimator = args["estimator"]
        # uwb = args["uwb"]
        optitrack = args["optitrack"]
        # trajectory = args["trajectory"]

        # Date
        date = datetime.today().strftime(r"%Y-%m-%d+%H:%M:%S")

        # Additional keywords
        # if keywords is not None:
        #     keywords = "+" + "+".join(keywords)
        # else:
        #     keywords = ""

        # Options
        if optitrack == "logging":
            options = f"optitracklog"
        elif optitrack == "state":
            options = f"optitrackstate"
        else:
            options = f""

        # Join
        name = "{}+{}.csv".format(date, options)
        fname = os.path.normpath(os.path.join(os.getcwd(), fileroot, name))
    return fname


def setup_logger():
    # Create directory if not there
    Path(args["fileroot"]).mkdir(exist_ok=True)
        
    # Create filename from options and date
    log_file = get_filename()

    print(f"Log location: {log_file}")


    # Logger setup
    logconfig = args["logconfig"]
    flogger = FileLogger(cf, logconfig, log_file)
    # flogger3 = FileLogger(cf, logconfig, log_file3)

    # Enable log configurations based on system setup:
    # Defaults
    # flogger.enableConfig("attitude")
    # flogger.enableConfig("gyros")
    # flogger.enableConfig("acc")
    flogger.enableConfig("state")
    flogger.enableConfig("whisker")
    flogger.enableConfig("whisker1")
    flogger.enableConfig("motor")
    flogger.enableConfig("otpos")
    flogger.enableConfig("orientation")

    # # UWB
    # if args["uwb"] == "twr":
    #     flogger.enableConfig("twr")
    # elif args["uwb"] == "tdoa":
    #     print("Needs custom TDoA logging in firmware!")
        # For instance, see here: https://github.com/Huizerd/crazyflie-firmware/blob/master/src/utils/src/tdoa/tdoaEngine.c
        # flogger.enableConfig("tdoa")
    # Flow
    if args["flow"]:
        flogger.enableConfig("laser")
    #     flogger.enableConfig("flow")
    # OptiTrack
    # if args["optitrack"] != "none":
    #     flogger.enableConfig("kalman")
    flogger.start()
    # flogger2.start()
    # flogger3.start()
    # # Estimator
    # if args["estimator"] == "kalman":
    #     flogger.enableConfig("kalman")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--fileroot", type=str, required=True)
    parser.add_argument("--logconfig", type=str, required=True)

    parser.add_argument(
        "--uwb", choices=["none", "twr", "tdoa"], type=str.lower, required=True
    )
    parser.add_argument("--flow", action="store_true")

    parser.add_argument(
        "--optitrack",
        choices=["none", "logging", "state"],
        type=str.lower,
        default="none",
    )

    parser.add_argument("--optitrack_id", type=int, default=None)
    parser.add_argument("--filename", type=str, default=None)
    args = vars(parser.parse_args())
    cflib.crtp.init_drivers()
    # Connect to the mocap system
    mocap_wrapper = MocapWrapper(rigid_body_name)
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf=scf.cf
        filelogger=setup_logger()
        # Set up a callback to handle data from the mocap system
        mocap_wrapper.on_pose = lambda pose: send_extpose_quat(cf, pose[0], pose[1], pose[2], pose[3])

        adjust_orientation_sensitivity(cf)
        activate_kalman_estimator(cf)
        reset_estimator(cf)
        move_linear_HC(scf,2.5)

    mocap_wrapper.close()