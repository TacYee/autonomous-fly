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
import numpy as np


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
with open('mlp_bs_32_lr_0.001_reg_0.001_do_0.1_rmse_9.0575_whisker1.txt', 'r') as f:
    lines = f.readlines()
    W1_1 = np.array([list(map(float, line.split())) for line in lines[1:33]])
    b1_1 = np.array([list(map(float, line.split())) for line in lines[35:67]]).flatten()
    W2_1 = np.array([list(map(float, line.split())) for line in lines[69:101]])
    b2_1 = np.array([list(map(float, line.split())) for line in lines[103:135]]).flatten()
    W3_1 = np.array([list(map(float, line.split())) for line in lines[137:169]])
    b3_1 = np.array([list(map(float, line.split())) for line in lines[171:203]]).flatten()
    W4_1 = np.array(list(map(float, lines[205].split())))
    b4_1 = np.array(list(map(float, lines[-1].split()))).flatten()
with open('mlp_bs_32_lr_0.001_reg_0.01_do_0.1_rmse_9.3832_whisker2.txt', 'r') as f:
    lines = f.readlines()
    W1_2 = np.array([list(map(float, line.split())) for line in lines[1:33]])
    b1_2 = np.array([list(map(float, line.split())) for line in lines[35:67]]).flatten()
    W2_2 = np.array([list(map(float, line.split())) for line in lines[69:101]])
    b2_2 = np.array([list(map(float, line.split())) for line in lines[103:135]]).flatten()
    W3_2 = np.array([list(map(float, line.split())) for line in lines[137:169]])
    b3_2 = np.array([list(map(float, line.split())) for line in lines[171:203]]).flatten()
    W4_2 = np.array(list(map(float, lines[205].split())))
    b4_2 = np.array(list(map(float, lines[-1].split()))).flatten()

def normalization(data, mean, std):
    normalized_data = (data - mean) / std
    return normalized_data

def relu(x):
    return np.maximum(0, x)

def mlp_inference(input_data, W1, b1, W2, b2, W3, b3, W4, b4):
    # 第一层前向传播
    z1 = np.dot(input_data, W1.T) + b1
    a1 = relu(z1)
    
    # 第二层前向传播
    z2 = np.dot(a1, W2.T) + b2
    a2 = relu(z2)
    
    # 第三层前向传播
    z3 = np.dot(a2, W3.T) + b3
    a3 = relu(z3)
    
    # 输出层前向传播
    output = np.dot(a3, W4.T) + b4
    
    return output


def dis_net1(whisker1, whisker2, whisker3):

    if whisker1 is None or whisker1 < 15:
        return 0
    else:
        input_data = np.array([whisker1, whisker2, whisker3])
        normalized_data = normalization(input_data, mean_1, std_1)
        output = mlp_inference(normalized_data, W1_1, b1_1, W2_1, b2_1, W3_1, b3_1, W4_1, b4_1)
    
        return output
    
def dis_net2(whisker1, whisker2, whisker3):

    if whisker2 is None or whisker2 < 15:
        return 0
    else:
        input_data = np.array([whisker1, whisker2, whisker3])
        normalized_data = normalization(input_data, mean_2, std_2)
        output = mlp_inference(normalized_data , W1_2, b1_2, W2_2, b2_2, W3_2, b3_2, W4_2, b4_2)
    
        return output


MIN_THRESHOLD = 30
MAX_THRESHOLD = 80

mean_1 = [ 54.20560548, -21.00775578, -14.01101664]
std_1 = [18.14239061,  8.27251843, 11.67656998]
mean_2 = [-1.64046324, 245.01769527, -85.01116184]
std_2 = [12.35055776, 101.31582844, 49.42377167]

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
                            motion_commander.start_turn_left(20)
                            time.sleep(0.01)
                        elif MAX_THRESHOLD > Distance1 > MIN_THRESHOLD and Distance2 > MAX_THRESHOLD:
                            motion_commander.start_turn_right(20)
                            time.sleep(0.01)
                        elif Distance1 < MIN_THRESHOLD and MAX_THRESHOLD > Distance2 > MIN_THRESHOLD:
                            motion_commander.start_turn_right(20)
                            time.sleep(0.01)
                        elif Distance1 > MAX_THRESHOLD and MAX_THRESHOLD > Distance2 > MIN_THRESHOLD:
                            motion_commander.start_turn_left(20)
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
                        file.write("timestamp,Distance1,Distance2\n")
                        # 写入数据
                        for timestamp, Distance1_value, Distance2_value in zip(timestamps, Distances1, Distances2):
                            file.write(f"{timestamp},{Distance1_value},{Distance2_value}\n")

                    print(f"数据已保存到文件: {file_name}")

                print('Demo terminated!')

# python3 whisker_hovering_collect_data.py --fileroot data/20240422 --logconfig logcfg.json
