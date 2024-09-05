import logging
import sys
import time
import os
import argparse
from datetime import datetime
from pathlib import Path

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
import whisker
from FileLogger import FileLogger
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')


# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)
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
        # optitrack = args["optitrack"]
        # trajectory = args["trajectory"]

        # Date
        date = datetime.today().strftime(r"%Y-%m-%d+%H:%M:%S")

        # Additional keywords
        # if keywords is not None:
        #     keywords = "+" + "+".join(keywords)
        # else:
        #     keywords = ""

        # Options
        # if optitrack == "logging":
        #     options = f"optitracklog"
        # elif optitrack == "state":
        #     options = f"optitrackstate"
        # else:
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
    # flogger.enableConfig("otpos")
    flogger.enableConfig("orientation")
    # flogger.enableConfig("WHISKER")

    # # UWB
    # if args["uwb"] == "twr":
    #     flogger.enableConfig("twr")
    # elif args["uwb"] == "tdoa":
    #     print("Needs custom TDoA logging in firmware!")
        # For instance, see here: https://github.com/Huizerd/crazyflie-firmware/blob/master/src/utils/src/tdoa/tdoaEngine.c
        # flogger.enableConfig("tdoa")
    # Flow
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

def is_touch(distance):

    if distance is None:
        return 0
    else:
        return distance 
    
# def start_motion(direction):
#     if direction == "forward":
#         motion_commander.start_linear_motion(0.2, 0, 0)
#     elif direction == "backward":
#         motion_commander.start_linear_motion(-0.2, 0, 0)
#     elif direction == "turn_left":
#         motion_commander.start_turn_left(10)
#     elif direction == "turn_right":
#         motion_commander.start_turn_right(10)

# def check_whiskers(MIN_THRESHOLD1, MAX_THRESHOLD1, MIN_THRESHOLD2, MAX_THRESHOLD2):
#     if MAX_THRESHOLD1 > WHISKER.whisker1_1> MIN_THRESHOLD1 and MAX_THRESHOLD2 > WHISKER.whisker2_2> MIN_THRESHOLD2:
#         start_motion("turn_right")
#     elif WHISKER.whisker1_1 > MAX_THRESHOLD1 and WHISKER.whisker2_2 > MAX_THRESHOLD2:
#         start_motion("turn_left")
#     elif WHISKER.whisker1_1 > MAX_THRESHOLD1 and WHISKER.whisker2_2 > MAX_THRESHOLD2:
#         start_motion("linear")
#     elif WHISKER.whisker1_1 < MIN_THRESHOLD1 and WHISKER.whisker2_2 < MIN_THRESHOLD2:
#         start_motion("linear")
#     elif WHISKER.whisker1_1 < MIN_THRESHOLD1:
#         start_motion("turn_right")
#     elif WHISKER.whisker2_2 < MIN_THRESHOLD2:
#         start_motion("turn_left")
#     elif WHISKER.whisker1_1 > MAX_THRESHOLD1:
#         start_motion("turn_left")
#     elif WHISKER.whisker2_2 > MAX_THRESHOLD2:
#         start_motion("turn_right")
#     else:
#         start_motion("linear")


MIN_THRESHOLD1 = 30
MAX_THRESHOLD1 = 100

MIN_THRESHOLD2 = 20
MAX_THRESHOLD2 = 80
whisker1_1_data = []
whisker2_1_data = []
timestamps = []
file_name = "whisker_data.csv"
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--fileroot", type=str, required=True)
    parser.add_argument("--logconfig", type=str, required=True)
    parser.add_argument("--filename", type=str, default=None)
    args = vars(parser.parse_args())
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        cf.platform.send_arming_request(True)
        with MotionCommander(scf) as motion_commander:
            time.sleep(3)
            with whisker.Whisker(scf) as WHISKER:
                filelogger=setup_logger()
                keep_flying = True
                try:

                    while keep_flying:
                        timestamps.append(time.time())
                        whisker1_1_data.append(is_touch(WHISKER.whisker1_1))
                        whisker2_1_data.append(is_touch(WHISKER.whisker2_1))
                        if MAX_THRESHOLD1 > is_touch(WHISKER.whisker1_1) > MIN_THRESHOLD1 and MAX_THRESHOLD2 > is_touch(WHISKER.whisker2_1)> MIN_THRESHOLD2:
                            motion_commander.start_linear_motion(0, -0.2, 0)
                            time.sleep(0.02)
                        elif MAX_THRESHOLD1 > is_touch(WHISKER.whisker1_1) > MIN_THRESHOLD1 and is_touch(WHISKER.whisker2_1) < MIN_THRESHOLD2:
                            motion_commander.start_turn_left(25)
                            time.sleep(0.02)
                        elif MAX_THRESHOLD1 > is_touch(WHISKER.whisker1_1) > MIN_THRESHOLD1 and is_touch(WHISKER.whisker2_1) > MAX_THRESHOLD2:
                            motion_commander.start_turn_right(25)
                            time.sleep(0.02)
                        elif is_touch(WHISKER.whisker1_1) < MIN_THRESHOLD1 and MAX_THRESHOLD2 > is_touch(WHISKER.whisker2_1) > MIN_THRESHOLD2:
                            motion_commander.start_turn_right(25)
                            time.sleep(0.02)
                        elif is_touch(WHISKER.whisker1_1) > MAX_THRESHOLD1 and MAX_THRESHOLD2 > is_touch(WHISKER.whisker2_1)> MIN_THRESHOLD2:
                            motion_commander.start_turn_left(25)
                            time.sleep(0.02)
                        elif is_touch(WHISKER.whisker1_1) > MAX_THRESHOLD1 and is_touch(WHISKER.whisker2_1) > MAX_THRESHOLD2 :
                            motion_commander.start_linear_motion(-0.1, 0, 0)
                            time.sleep(0.02)
                        else :
                            motion_commander.start_linear_motion(0.2, 0, 0)
                            time.sleep(0.02)
                except KeyboardInterrupt:
                    with open(file_name, 'w') as file:
                        # 写入表头
                        file.write("timestamp,whisker1_1,whisker2_1\n")
                        # 写入数据
                        for timestamp, whisker1_1_value, whisker2_1_value in zip(timestamps, whisker1_1_data, whisker2_1_data):
                            file.write(f"{timestamp},{whisker1_1_value},{whisker2_1_value}\n")

                    print(f"file saved!: {file_name}")

                print('Demo terminated!')

# python3 whisker_hovering_collect_data.py --fileroot data/20240422 --logconfig logcfg.json
