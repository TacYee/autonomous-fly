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

import torch
from model import *

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EF')


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

model1 = MLP(3, 32, 1, 0.1)
model2 = MLP(3, 32, 1, 0.1)
best_model1_path='mlp_bs_32_lr_0.001_reg_0.001_do_0.1_rmse_9.0575_whisker1.pt'
best_model2_path='mlp_bs_32_lr_0.001_reg_0.01_do_0.1_rmse_9.3832_whisker2.pt'
model1.load_state_dict(torch.load(best_model1_path))
model2.load_state_dict(torch.load(best_model2_path))
model1.eval()
model2.eval()

def dis_net1(whisker1, whisker2, whisker3):

    if whisker1 is None or whisker1 < 15:
        return 0
    else:
        input_data = torch.tensor([[whisker1, whisker2, whisker3]], dtype=torch.float32)
        with torch.no_grad():
            output = model1(input_data)
    
        return output
    
def dis_net2(whisker1, whisker2, whisker3):

    if whisker2 is None or whisker2 < 15:
        return 0
    else:
        input_data = torch.tensor([[whisker1, whisker2, whisker3]], dtype=torch.float32)
        with torch.no_grad():
            output = model2(input_data)
    
        return output


MIN_THRESHOLD = 30
MAX_THRESHOLD = 50
Distances1 = []
Distances2 = []
timestamps = []
file_name = "whisker_data_MLP.csv"
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
        with MotionCommander(scf) as motion_commander:
            time.sleep(3)
            with whisker.Whisker(scf) as WHISKER:
                filelogger=setup_logger()
                keep_flying = True
                try:
                    while keep_flying:
                        timestamps.append(time.time())
                        Distance1 = dis_net1(WHISKER.whisker1_1,WHISKER.whisker1_2,WHISKER.whisker1_3) 
                        Distance2 = dis_net2(WHISKER.whisker2_1,WHISKER.whisker2_2,WHISKER.whisker2_3)
                        Distances1.append(Distance1)
                        Distances2.append(Distance2)
                        if MAX_THRESHOLD > Distance1 > MIN_THRESHOLD and MAX_THRESHOLD > Distance2 > MIN_THRESHOLD:
                            motion_commander.start_linear_motion(0, -0.2, 0)
                            time.sleep(0.01)
                        elif MAX_THRESHOLD > Distance1 > MIN_THRESHOLD and Distance2 < MIN_THRESHOLD:
                            motion_commander.start_turn_left(10)
                            time.sleep(0.01)
                        elif MAX_THRESHOLD > Distance1 > MIN_THRESHOLD and Distance2 > MAX_THRESHOLD:
                            motion_commander.start_turn_right(10)
                            time.sleep(0.01)
                        elif Distance1 < MIN_THRESHOLD and MAX_THRESHOLD > Distance2 > MIN_THRESHOLD:
                            motion_commander.start_turn_right(10)
                            time.sleep(0.01)
                        elif Distance1 > MAX_THRESHOLD and MAX_THRESHOLD > Distance2 > MIN_THRESHOLD:
                            motion_commander.start_turn_left(10)
                            time.sleep(0.01)
                        elif Distance1 > MAX_THRESHOLD and Distance2 > MAX_THRESHOLD :
                            motion_commander.start_linear_motion(-0.2, 0, 0)
                            time.sleep(0.01)
                        else :
                            motion_commander.start_linear_motion(0.2, 0, 0)
                            time.sleep(0.01)
                except KeyboardInterrupt:
                    with open(file_name, 'w') as file:
                        # 写入表头
                        file.write("timestamp,whisker1_1,whisker2_2\n")
                        # 写入数据
                        for timestamp, Distance1_value, Distance2_value in zip(timestamps, Distance1, Distance2):
                            file.write(f"{timestamp},{Distance1_value},{Distance2_value}\n")

                    print(f"数据已保存到文件: {file_name}")

                print('Demo terminated!')

# python3 whisker_hovering_collect_data.py --fileroot data/20240422 --logconfig logcfg.json
