import numpy as np
import matplotlib.pyplot as plt
import time
import re
    
GRAVITY=9.80665
###############################################################################################################
class ConstantAccelerationKalmanFilter:
    def __init__(self, process_var, initial_measurement_var, initial_time=time.time()):
        """
        process_var: Process noise variance (model uncertainty)
        initial_measurement_var: Initial measurement noise variance (different for each dimension)
        """
        self.x = np.array([1,0,0])  # Initial state [px, vx, ax]
        
        # State transition matrix (updated dynamically)
        self.F = np.eye(3)

        # Position Measurement matrix
        self.Hpos = np.array([[1, 0, 0]])
        # Acceleration Measurement matrix
        self.Hacc = np.array([[0, 0, 1]])
        
        # Process covariance matrix (model uncertainty)
        self.Q_base = np.array([[1/4, 1/2, 1/6], [1/2, 1, 1/2], [1/6, 1/2, 1]]) * process_var
        
        # Measurement covariance matrix 
        self.Rpos = np.diag([initial_measurement_var[0]])
        self.Racc = np.diag([initial_measurement_var[1]])
        
        # Initial estimation error covariance matrix
        self.P = np.eye(3)
        self.measure_time = initial_time
        
    def _reset(self, pos, vel=0, acc=0, time_=time.time()):
        """ Reset the filter with a new state """
        self.x = np.array([pos, vel, acc])
        self.P = np.eye(3)
        self.measure_time = time_
                
    def _predict(self, dt=None, time_=time.time()):
        """ Predict the next state and uncertainty based on variable time step dt """
        if dt is None:
            dt = time_ - self.measure_time
        F_block = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
        self.F = F_block
        self.Q = self.Q_base * (dt**2)
        
        self.x = self.F @ self.x  # State prediction
        self.P = self.F @ self.P @ self.F.T + self.Q  # Covariance prediction
        
    def _updatePos(self, value, measurement_var, Hmat, Rmat, time_=time.time()):
        """ Update state based on measurement value with time-varying noise """
        Rmat = np.diag(measurement_var)  # Update measurement covariance dynamically
        y = value - (Hmat @ self.x)  # Measurement residual
        S = Hmat @ self.P @ Hmat.T + Rmat  # Residual covariance
        K = self.P @ Hmat.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + (K @ y)  # Updated state estimate
        self.P = (np.eye(3) - K @ Hmat) @ self.P @ (np.eye(3) - K @ Hmat).T + K @ Rmat @ K.T  # Full covariance update
        self.measure_time = time_
        
    def updateWithPosition(self, pos, measurement_var, time_=time.time()):
        self._predict(time_=time_)
        self._updatePos(value=[pos], measurement_var=[measurement_var], Hmat=self.Hpos, Rmat=self.Rpos, time_=time_)

    def updateWithAcceleration(self, acc, measurement_var, time_=time.time()):
        self._predict(time_=time_)
        self._updatePos(value=acc, measurement_var=[measurement_var], Hmat=self.Hacc, Rmat=self.Racc, time_=time_)

    def get_state(self, dt=None, time_=time.time()):
        """ Return current estimated position, velocity, and acceleration """
        if dt is None:
            dt = time_ - self.measure_time
        # dt is the time from the last measurement update
        F_block = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
        F = F_block
        x = F @ self.x
        pos = x[0]
        vel = x[1]
        acc = x[2]
        return self.x, (pos, vel, acc)
###############################################################################################################
def parse_mavlink_log_line(log_file_path):
    # Open connection to log file
    log_file = open(log_file_path, 'r')

    parsed_data = {}

    for line in log_file:
        match = re.match(r"\{(.*)\}", line.strip())
        if not match:
            continue

        content = match.group(1)
        msg_dict = {}
        

        # Split on commas not within strings (basic split for this format)
        items = content.split(", ")
        
        for item in items:
            if ": " in item:
                key, value = item.split(": ", 1)
                key = key.strip(); key = key.replace("'", "")
                value = value.strip(); value = value.replace("'", "")
                # Convert to float or int if possible
                try:
                    if '.' in value or 'e' in value.lower():
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # leave as string if cannot convert
                msg_dict[key] = value

        if msg_dict['mavpackettype'] in parsed_data:
            # Check if the message type already exists in parsed_data
            for key in msg_dict:
                parsed_data[msg_dict['mavpackettype']][key].append(msg_dict[key])
        else:
            parsed_data[msg_dict['mavpackettype']] = {}
            for key in msg_dict:
                if key not in parsed_data[msg_dict['mavpackettype']]:
                    parsed_data[msg_dict['mavpackettype']][key] = [msg_dict[key]]
        
    for key in parsed_data:
        parsed_data[key].pop('mavpackettype', None)        # Remove the 'mavpackettype' key from the dictionary
    return parsed_data
    return parsed_data
###############################################################################################################
def pressure_to_altitude(P, T, P0=101325):
    """
    Converts barometric pressure (Pa) and temperature (K) to altitude (m).
    
    Parameters:
        P (float): Pressure in Pascals
        T (float): Temperature in Kelvin
        P0 (float): Sea level standard pressure in Pascals (default is 101325 Pa)
        
    Returns:
        h (float): Altitude in meters
    """
    # Constants
    R = 287.05      # Specific gas constant for dry air, J/(kg·K)
    g = 9.80665     # Acceleration due to gravity, m/s²
    
    # Barometric formula for constant temperature
    h = - (R * T / g) * np.log(P / P0)
    
    return h
###############################################################################################################
def pressure_to_altitudeJacob(P, T, P0=101325):
    G=-9.80665
    MOLECULAR_WEIGHT_OF_AIR = 0.0289644
    UNIVERSAL_GAS_CONSTANT= 8.31446261815324

    STANDARD_TEMP_LAPSE_RATE_K_PER_KM = -0.0065
    
    exp = (UNIVERSAL_GAS_CONSTANT * STANDARD_TEMP_LAPSE_RATE_K_PER_KM )/ (GRAVITY*MOLECULAR_WEIGHT_OF_AIR) 
    h = (T/STANDARD_TEMP_LAPSE_RATE_K_PER_KM) * (pow((P0/P), exp)-1)
            
    return h
###############################################################################################################

if __name__ == '__main__':
    
    log_file_path = '/home/valentin/Projects/GambitonBiut/Jacob/records/15-15-10/mavlink.txt'
    parsed_data = parse_mavlink_log_line(log_file_path)
    # Simulate noisy 3D position measurements with variable time steps and measurement noise
    

    baroData = [];     baroData_ = {}
    baroData_['time'] = np.array(parsed_data['SCALED_PRESSURE']['time_boot_ms'])/1e3; startTime = baroData_['time'][0]; baroData_['time'] -= startTime
    baroData_['pressure'] = np.array(parsed_data['SCALED_PRESSURE']['press_abs'])*100
    baroData_['temperature'] = np.array(parsed_data['SCALED_PRESSURE']['temperature'])   
    baroData.append(baroData_)
    
    baroData_['time'] = np.array(parsed_data['SCALED_PRESSURE2']['time_boot_ms'])/1e3;    baroData_['time'] -= startTime
    baroData_['pressure'] = np.array(parsed_data['SCALED_PRESSURE2']['press_abs'])*100
    baroData.append(baroData_)    

    imu = [];           imu_ = {}
    imu_['time'] = np.array(parsed_data['RAW_IMU']['time_usec'])/1e6;    imu_['time'] -= startTime
    imu_['zacc'] = np.array(parsed_data['RAW_IMU']['zacc'])*GRAVITY/1000 + GRAVITY
    imu.append(imu_)

    imu_['time'] = np.array(parsed_data['SCALED_IMU2']['time_boot_ms'])/1e3;    imu_['time'] -= startTime
    imu_['zacc'] = np.array(parsed_data['SCALED_IMU2']['zacc'])*GRAVITY/1000 + GRAVITY
    imu.append(imu_)

    imu_['time'] = np.array(parsed_data['SCALED_IMU3']['time_boot_ms'])/1e3;    imu_['time'] -= startTime
    imu_['zacc'] = np.array(parsed_data['SCALED_IMU3']['zacc'])*GRAVITY/1000 + GRAVITY
    imu.append(imu_)
    
    pos = {}
    pos['time'] = np.array(parsed_data['GLOBAL_POSITION_INT']['time_boot_ms'])/1e3
    pos['time'] -= startTime
    pos['alt'] = np.array(parsed_data['GLOBAL_POSITION_INT']['alt'])/1000
    pos['alt'] -= pos['alt'][0]
    
    posRaw = {}
    posRaw['time'] = np.array(parsed_data['GPS_RAW_INT']['time_usec'])/1e6
    posRaw['time'] -= startTime
    posRaw['alt'] = np.array(parsed_data['GPS_RAW_INT']['alt'])/1000
    posRaw['alt'] -= posRaw['alt'][0]
    

    SCALE_BARO2HEIGHT = 0.12  # not exactly correct, and increase unnecessary error
    for baroData_ in baroData:
        inds = np.where(baroData_['pressure'] > 120000)
        # baroData_['pressure'][inds] = baroData_['pressure'][inds+1]
        baroData_['heightFunc'] = pressure_to_altitude(baroData_['pressure'], T=300, P0=baroData_['pressure'][0])
        baroData_['heightFuncJacob'] = pressure_to_altitudeJacob(baroData_['pressure'], T=300, P0=baroData_['pressure'][0])
        baroData_['heightSimple'] = -(baroData_['pressure']-baroData_['pressure'][0])*SCALE_BARO2HEIGHT    
    
    for imu_ in imu:
        dt = np.diff(imu_['time']); dt = np.concatenate(([0], dt))
        vz = np.cumsum(imu_['zacc']*dt);   
        pz = np.cumsum(imu_['zacc']*dt);   
        imu_['vz'] = vz
        imu_['pz'] = pz
        
    # plt.figure(1)
    # plt.subplot(2, 1, 1)
    # plt.plot(baroData[0]['time'], baroData[0]['heightFunc'], 'x', label='Barometric[0] Height Function')
    # # plt.plot(baroData[1]['time'], baroData[1]['heightFunc'], '^', label='Barometric[1] Height Function')
    # plt.plot(baroData[0]['time'], baroData[0]['heightFuncJacob'], '+', label='Barometric Jacob Height Function')
    # # plt.plot(baroData[0]['time'], baroData[0]['heightSimple'], 'o',label='Barometric Height Simple')
    # # plt.plot(imu[0]['time'], imu[1]['pz'], '+', label='IMU Height RAW')
    # # plt.plot(imu[1]['time'], imu[1]['pz'], '>', label='IMU Height SCALED')
    # plt.plot(posRaw['time'], posRaw['alt'], '.', label='GPS Altitude raw')
    # plt.plot(pos['time'], pos['alt'], '.', label='GPS Altitude')
    # plt.grid(True)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Height (m)')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(baroData[0]['time'], baroData[0]['pressure'], 'x', label='Barometric Pressure')
    # plt.grid(True)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Pressure (Pa)')
    # plt.legend()
    # plt.show()
    # i=1         
    
    cakf = ConstantAccelerationKalmanFilter(process_var=1.0, initial_measurement_var=[1.1,1.2], initial_time=0)

    true_position = pos['alt']
    estimated_position = np.zeros(len(pos['alt']))

    positionsca = []; positionscv = []
    velocitiesca = []; velocitiescv = []
    accelerationsca = []
    measurements = []
    true_positions = []
    time_steps = pos['time'] #np.cumsum(np.random.uniform(0.5, 1.5, size=50))  # Variable time intervals
    
    baroInd = 0
    imuInd = 0
    trueInd = 0
    
    while not (trueInd==len(pos['time']) or imuInd==len(imu[0]['time']) or baroInd==len(baroData[0]['time'])):
        curBaroTime = baroData[0]['time'][baroInd]
        curImuTime = imu[0]['time'][imuInd]
        curTrueTime = pos['time'][trueInd]
        
        if curBaroTime <= curImuTime and curBaroTime <= curTrueTime:
            measurement = baroData[0]['heightFunc'][baroInd] 
            cakf.updateWithPosition(pos=measurement, time_=curBaroTime, measurement_var=1)
            baroInd += 1
        elif curImuTime < curBaroTime and curImuTime <= curTrueTime:
            measurement = imu[0]['zacc'][imuInd] 
            cakf.updateWithAcceleration(acc=measurement, time_=curImuTime, measurement_var=1)
            imuInd += 1
        elif curTrueTime < curBaroTime and curTrueTime < curImuTime:
            estimated_position[trueInd] = cakf.get_state(time_=curTrueTime)[1][0]
            trueInd += 1
               
            cakfState,_ = cakf.get_state(dt=0)
            # Extract position, velocity, and acceleration from the state
            posca, velca, accca = cakfState[0], cakfState[1], cakfState[2]
            positionsca.append(posca)
            velocitiesca.append(velca)
            accelerationsca.append(accca)

    positionsca = np.array(positionsca); velocitiesca = np.array(velocitiesca); accelerationsca = np.array(accelerationsca)
    # Plot results
    dim_labels = ['X', 'Y', 'Z']

    plt.figure(10)
    plt.subplot(2,1,1)
    plt.plot(time_steps, positionsca, label='Estimated Position', linestyle='-')
    plt.scatter(baroData[0]['time'], baroData[0]['heightFunc'], label='Measurements', color='red', s=10)
    plt.plot(pos['time'], pos['alt'], label='True Position', linestyle='solid')
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(time_steps, accelerationsca, label='Estimated Acceleration', linestyle='-')
    plt.scatter(imu[0]['time'], imu[0]['zacc'], label='Measurements', color='red', s=10)
    plt.xlabel("Time")
    plt.ylabel("Acceleration")
    plt.legend()
    
    plt.show()
    i=1