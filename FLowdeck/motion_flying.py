import argparse
import logging
import sys
import time
from threading import Event
from pathlib import Path
from datetime import datetime
import os

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from FileLogger import FileLogger

URI = uri_helper.uri_from_env(default='radio://0/80/2M/CFE7E7E701')

DEFAULT_HEIGHT = 0.5

deck_attached_event = Event()

logging.basicConfig(level=logging.ERROR)

position_estimate = [0, 0]

def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()

def move_linear_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(10)
        mc.forward(0.5)
        time.sleep(10)
        mc.back(0.5)
        time.sleep(10)
        mc.stop

def move_linear_HC(scf):

    flight_time = 4

    commander = scf.cf.high_level_commander

    commander.takeoff(0.5, 2)
    time.sleep(3)

    commander.go_to(0.5, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(0, -0.5, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(-0.5, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(0, 0, -0.5, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(0, 0, -0.2, 0, flight_time, relative=True)

    commander.land(0, 2.0)

    commander.stop()

# def log_pos_callback(timestamp, data, logconf):
#     print(data)
#     global position_estimate
#     position_estimate[0] = data['stateEstimate.x']
#     position_estimate[1] = data['stateEstimate.y']

def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

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

    # Enable log configurations based on system setup:
    # Defaults
    flogger.enableConfig("attitude")
    flogger.enableConfig("gyros")
    flogger.enableConfig("acc")
    flogger.enableConfig("state")
    flogger.enableConfig("whisker")
    flogger.enableConfig("motor")

    # UWB
    if args["uwb"] == "twr":
        flogger.enableConfig("twr")
    elif args["uwb"] == "tdoa":
        print("Needs custom TDoA logging in firmware!")
        # For instance, see here: https://github.com/Huizerd/crazyflie-firmware/blob/master/src/utils/src/tdoa/tdoaEngine.c
        # flogger.enableConfig("tdoa")
    # Flow
    if args["flow"]:
        flogger.enableConfig("laser")
        flogger.enableConfig("flow")
    # OptiTrack
    if args["optitrack"] != "none":
        flogger.enableConfig("otpos")
        flogger.enableConfig("otatt")
    flogger.start()
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
    cf=Crazyflie(rw_cache='./cache')

    with SyncCrazyflie(URI, cf) as scf:

        filelogger=setup_logger()

        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        move_linear_HC(scf)


