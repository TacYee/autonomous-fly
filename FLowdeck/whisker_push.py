import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
import whisker
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/CFE7E7E701')

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

def model_linear_drift(batch_data):
    x = np.arange(100).reshape(-1, 1) 
    y = batch_data
    model = LinearRegression().fit(x, y)
    slope = model.coef_[0]
    intercept = model.intercept_

def is_touch(distance):
    threshold = 10  

    if distance is None:
        return False
    else:
        return distance > threshold


if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        with MotionCommander(scf) as motion_commander:
            with whisker(scf) as WHISKER:
                time.sleep(2)
                keep_flying = True

                while keep_flying:
                    VELOCITY = 0.2
                    velocity_x = 0.0
                    velocity_y = 0.0

                    if is_touch(WHISKER.whisker1_2) or is_touch(WHISKER.whisker2_2):
                        velocity_x -= VELOCITY
                        motion_commander.start_linear_motion(
                            velocity_x, velocity_y, 0)
                        time.sleep(1)

                    time.sleep(0.02)

            print('Demo terminated!')
