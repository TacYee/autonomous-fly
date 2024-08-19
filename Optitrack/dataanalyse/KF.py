import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = initial_state  # 状态估计
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵

    def predict(self):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre
        
        # 预测协方差矩阵：P_k^- = P_k-1 + Q
        self.P_minus = self.P + self.Q

    def update(self, z):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return self.x_pre
    

class KalmanFilterUA:
    def __init__(self, initial_state, initial_covariance, process_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = initial_state  # 状态估计
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = None  # 测量噪声协方差矩阵

    def predict(self):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre
        
        # 预测协方差矩阵：P_k^- = P_k-1 + Q
        self.P_minus = self.P + self.Q

    def update(self, z, measurement_noise):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        self.R  =measurement_noise
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return self.x_pre
    
class KalmanFilterFLAT:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise, measurement_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 234 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = measurement_noise  # 测量噪声协方差矩阵

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre + (p_x - self.p_x_last) * np.sin(np.deg2rad(self.Θ_last)) + (self.p_y_last - p_y) * np.cos(np.deg2rad(self.Θ_last))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus = self.P + self.Q
        self.Θ_minus = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre
        
class KalmanFilterUAFLAT:
    def __init__(self, initial_state, initial_position, initial_yaw, initial_covariance, process_noise):
        """
        初始化卡尔曼滤波器。
        
        :param initial_state: 初始状态估计值，向量 x_hat_0
        :param initial_covariance: 初始协方差矩阵 P_0
        :param process_noise: 过程噪声协方差矩阵 Q
        :param measurement_noise: 测量噪声协方差矩阵 R
        """
        self.x_pre = 234 - initial_state  # 状态估计
        self.p_x_last = -(initial_position[1] * 1000)
        self.p_y_last = initial_position[0] * 1000
        self.Θ_last = initial_yaw
        self.P = initial_covariance  # 协方差矩阵
        self.Q = process_noise  # 过程噪声协方差矩阵
        self.R = None  # 测量噪声协方差矩阵

    def predict(self, position, yaw):
        """
        预测步骤：预测下一个状态和协方差矩阵。
        """
        p_x = -(position[1] * 1000)
        p_y = position[0] * 1000
        # 预测状态：x_hat_k^- = x_hat_k-1
        self.x_pre_minus = self.x_pre + (p_x - self.p_x_last) * np.sin(np.deg2rad(self.Θ_last)) + (self.p_y_last - p_y) * np.cos(np.deg2rad(self.Θ_last))
        
        # 预测协方差矩阵：P_k^- = P_k-1 + 
        self.P_minus = self.P + self.Q
        self.Θ_minus = yaw
        self.p_x_last = p_x
        self.p_y_last = p_y

    def update(self, z, measurement_noise):
        """
        测量更新步骤：根据新的测量值更新状态估计和协方差矩阵。
        
        :param z: 当前测量值
        """
        self.R  = measurement_noise
        # 卡尔曼增益：K_k = P_k^- * (P_k^- + R_k)^-1
        self.K = self.P_minus @ np.linalg.inv(self.P_minus + self.R)
        
        # 更新状态估计：x_hat_k = x_hat_k^- + K_k * (z_k - x_hat_k^-)
        self.x_pre = self.x_pre_minus + self.K @ (z - self.x_pre_minus)
        
        # 更新协方差矩阵：P_k = (I - K_k) * P_k^-
        I = np.eye(self.P.shape[0])  # 单位矩阵
        self.P = (I - self.K) @ self.P_minus

    def get_current_estimate(self):
        """
        获取当前状态估计。
        
        :return: 当前状态估计值
        """
        return 234 - self.x_pre