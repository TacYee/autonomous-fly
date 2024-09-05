import logging
import sys
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
import whisker
from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7EF')

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


def is_touch_1(distance):
    threshold = 40  

    if distance is None:
        return False
    else:
        return distance > threshold
    
def is_touch_2(distance):
    threshold = 30  

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
            print(1)
            with whisker.Whisker(scf) as WHISKER:
                print(1)
                keep_flying = True
                while keep_flying:
                    VELOCITY = 0.2
                    velocity_x = 0.0
                    velocity_y = 0.0

                    if is_touch_1(WHISKER.whisker1_1) or is_touch_2(WHISKER.whisker2_1):
                        velocity_x -= VELOCITY

                    motion_commander.start_linear_motion(
                        velocity_x, velocity_y, 0)
                    
                    time.sleep(0.2)
                    print(WHISKER.whisker1_1, WHISKER.whisker2_1)


            print('Demo terminated!')
