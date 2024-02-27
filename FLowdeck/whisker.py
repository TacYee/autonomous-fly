from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from scipy import signal

class Whisker:
    WHISKER1_1 = 'Whisker.Barometer1_1'
    WHISKER1_2 = 'Whisker.Barometer1_2'
    WHISKER1_3 = 'Whisker.Barometer1_3'
    WHISKER2_1 = 'Whisker1.Barometer2_1'
    WHISKER2_2 = 'Whisker1.Barometer2_2'
    WHISKER2_3 = 'Whisker1.Barometer2_3'

    def __init__(self, crazyflie, rate_ms=20):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        # self._whisker1_1 = None
        self._whisker1_2 = None
        # self._whisker1_3 = None
        # self._whisker2_1 = None
        self._whisker2_2 = None
        # self._whisker2_3 = None
        self._b, self._a = self._calculate_filter_coefficients()
        self._zi1 = None
        self._zi2 = None

    def _calculate_filter_coefficients(self):
        """
        Calculate the filter coefficients for a lowpass filter.
        """
        high_freq = 3
        fs = 50
        b, a = signal.butter(1, high_freq / (0.5 * fs), 'lowpass')
        return b, a


    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Whisker1', rate_ms)
        # log_config.add_variable(self.WHISKER1_1)
        log_config.add_variable(self.WHISKER1_2)
        # log_config.add_variable(self.WHISKER1_3)
        # log_config.add_variable(self.WHISKER2_1)
        log_config.add_variable(self.WHISKER2_2)
        # log_config.add_variable(self.WHISKER2_3)

        log_config.data_received_cb.add_callback(self._data_received)

        return log_config
    
        
    def _apply_lowpass_filter_realtime(self, data_point, zi):
        # Initialize self._zi if it is None
        if zi is None:
            zi = signal.lfilter_zi(self._b, self._a) * data_point
        
        # Apply lowpass filter
        filtered_data_point, zi = signal.lfilter(self._b, self._a, [data_point], zi=zi)
        return filtered_data_point, zi

    def _data_received(self, timestamp, data, logconf):
        self._whisker1_2, self._zi1 = self._apply_lowpass_filter_realtime(data[self.WHISKER1_2],self._zi1)
        self._whisker2_2, self._zi2 = self._apply_lowpass_filter_realtime(data[self.WHISKER2_2],self._zi2)

    def start(self):
        self._cf.log.add_config(self._log_config)
        self._log_config.start()

    def stop(self):
        self._log_config.delete()


    @property
    def whisker1_1(self):
        return self._whisker1_1

    @property
    def whisker1_2(self):
        return self._whisker1_2

    @property
    def whisker1_3(self):
        return self._whisker1_3

    @property
    def whisker2_1(self):
        return self._whisker2_1

    @property
    def whisker2_2(self):
        return self._whisker2_2

    @property
    def whisker2_3(self):
        return self._whisker2_3
