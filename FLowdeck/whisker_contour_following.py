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


def is_touch(distance):
    threshold = 20  

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
          
                keep_flying = True
                while keep_flying:

                    if is_touch(WHISKER.whisker1_2) or is_touch(WHISKER.whisker2_2):

                        motion_commander.start_linear_motion(
                            0, -0.2, 0)
                        time.sleep(0.01)
                    else:

                        motion_commander.start_linear_motion(
                            0.15, 0, 0)
                    
                        time.sleep(0.01)


            print('Demo terminated!')
