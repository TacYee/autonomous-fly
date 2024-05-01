from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from scipy import signal
import numpy as np

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

        self._whisker1_1 = None
        self._whisker1_2 = None
        self._whisker1_3 = None
        self._whisker2_1 = None
        self._whisker2_2 = None
        self._whisker2_3 = None
        self._b, self._a = self._calculate_filter_coefficients()
        self._zi1_1 = None
        self._zi1_2 = None
        self._zi1_3 = None
        self._zi2_1 = None
        self._zi2_2 = None
        self._zi2_3 = None
        self._first_100_data_1_1 = []
        self._first_100_data_1_2 = []
        self._first_100_data_1_3 = []
        self._first_100_data_2_1 = []
        self._first_100_data_2_2 = []
        self._first_100_data_2_3 = [] 
        self._slope_1_1 = None  # save slope
        self._intercept_1_1 = None  # save intercept
        self._slope_1_2 = None
        self._intercept_1_2 = None
        self._slope_1_3 = None
        self._intercept_1_3 = None
        self._slope_2_1 = None
        self._intercept_2_1 = None
        self._slope_2_2 = None
        self._intercept_2_2 = None
        self._slope_2_3 = None
        self._intercept_2_3 = None
        self._time_stamp = 0

    
    def _linear_fit(self, y):
        X =  np.column_stack((np.arange(100).reshape(-1, 1), np.ones_like(np.arange(100).reshape(-1, 1)))) # Add column of ones for intercept
        coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        slope = coefficients[0]
        intercept = coefficients[1]
        return slope, intercept
        
    def _calculate_filter_coefficients(self):
        """
        Calculate the filter coefficients for a lowpass filter.
        """
        high_freq = 1
        low_freq = 0.05
        fs = 50
        b, a = signal.butter(1, [low_freq / (0.5 * fs), high_freq / (0.5 * fs)], 'bandpass')
        return b, a


    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Whisker1', rate_ms)
        log_config.add_variable(self.WHISKER1_1)
        log_config.add_variable(self.WHISKER1_2)
        log_config.add_variable(self.WHISKER1_3)
        log_config.add_variable(self.WHISKER2_1)
        log_config.add_variable(self.WHISKER2_2)
        log_config.add_variable(self.WHISKER2_3)

        log_config.data_received_cb.add_callback(self._data_received)

        return log_config
    
        
    def _apply_bandpass_filter_realtime(self, residuals, zi):
        # Initialize self._zi if it is None
        if zi is None:
            zi = signal.lfilter_zi(self._b, self._a) * residuals
        
        # Apply 1st-order bandpass filter
        filtered_data_point, zi = signal.lfilter(self._b, self._a, [residuals], zi=zi)
        return filtered_data_point[0], zi

    def _data_received(self, timestamp, data, logconf):
        if self._slope_1_1 is None:
            self._initialize_linear_model(data[self.WHISKER1_1], data[self.WHISKER1_2], data[self.WHISKER1_3], data[self.WHISKER2_1], data[self.WHISKER2_2], data[self.WHISKER2_3])
        else:
            self._process_data_point(data[self.WHISKER1_1], data[self.WHISKER1_2], data[self.WHISKER1_3], data[self.WHISKER2_1], data[self.WHISKER2_2], data[self.WHISKER2_3])

    def _initialize_linear_model(self, data_point_1_1, data_point_1_2, data_point_1_3, data_point_2_1, data_point_2_2, data_point_2_3):
        self._first_100_data_1_1.append(data_point_1_1)
        self._first_100_data_1_2.append(data_point_1_2)
        self._first_100_data_1_3.append(data_point_1_3)
        self._first_100_data_2_1.append(data_point_2_1)
        self._first_100_data_2_2.append(data_point_2_2)
        self._first_100_data_2_3.append(data_point_2_3)
        self._time_stamp += 1

        if self._time_stamp == 100:
            self._slope_1_1, self._intercept_1_1 = self._linear_fit(self._first_100_data_1_1)
            self._slope_1_2, self._intercept_1_2 = self._linear_fit(self._first_100_data_1_2)
            self._slope_1_3, self._intercept_1_3 = self._linear_fit(self._first_100_data_1_3)
            self._slope_2_1, self._intercept_2_1 = self._linear_fit(self._first_100_data_2_1)
            self._slope_2_2, self._intercept_2_2 = self._linear_fit(self._first_100_data_2_2)
            self._slope_2_3, self._intercept_2_3 = self._linear_fit(self._first_100_data_2_3)

    def _process_data_point(self, data_point_1_1, data_point_1_2, data_point_1_3, data_point_2_1, data_point_2_2, data_point_2_3):
        residuals_1_1 = data_point_1_1 - (self._slope_1_1 * self._time_stamp + self._intercept_1_1)
        residuals_1_2 = data_point_1_2 - (self._slope_1_2 * self._time_stamp + self._intercept_1_2)
        residuals_1_3 = data_point_1_3 - (self._slope_1_3 * self._time_stamp + self._intercept_1_3)
        residuals_2_1 = data_point_2_1 - (self._slope_2_1 * self._time_stamp + self._intercept_2_1)
        residuals_2_2 = data_point_2_2 - (self._slope_2_2 * self._time_stamp + self._intercept_2_2)
        residuals_2_3 = data_point_2_3 - (self._slope_2_3 * self._time_stamp + self._intercept_2_3)
        self._whisker1_1, self._zi1_1 = self._apply_bandpass_filter_realtime(residuals_1_1, self._zi1_1)
        self._whisker1_2, self._zi1_2 = self._apply_bandpass_filter_realtime(residuals_1_2, self._zi1_2)
        self._whisker1_3, self._zi1_3 = self._apply_bandpass_filter_realtime(residuals_1_3, self._zi1_3)
        self._whisker2_1, self._zi2_1 = self._apply_bandpass_filter_realtime(residuals_2_1, self._zi2_1)
        self._whisker2_2, self._zi2_2 = self._apply_bandpass_filter_realtime(residuals_2_2, self._zi2_2)
        self._whisker2_3, self._zi2_3 = self._apply_bandpass_filter_realtime(residuals_2_3, self._zi2_3)

        self._time_stamp += 1


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
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()



