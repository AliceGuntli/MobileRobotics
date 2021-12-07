import numpy as np

# https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/

# Supress scientific notation when printing NumPy arrays
np.set_printoptions(precision=3, suppress=True)

# A matrix
# Expresses how the state of the system [x,y,angle] changes from k-1 to k when no control command is executed.
# For this case, A is the identity matrix. a robot on wheels only drives when the wheels are told to turn.
A_k_minus_1 = np.array([[1.0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 1.0]])

# Noise applied to the forward kinematics => to be tweaked
process_noise_v_k_minus_1 = np.array([0.1, 0.1, 0.1])

# State model noise covariance matrix Q_k, first choice identity :
Q_k = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Measurement matrix H_k, every state is observable so H is 3x3
H_k = np.array([[1.0, 0, 0],
                [0, 1.0, 0],
                [0, 0, 1.0]])

# Sensor measurement noise covariance matrix R_k => to be tweaked, lower values = more confidence in measurements
R_k = np.array([[0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5]])

# Sensor noise => to be tweaked
sensor_noise_w_k = np.array([0.01, 0.01, 0.01])


def getB(angle, deltak):
    """
    Calculates and returns the B matrix
    Expresses how the state of the system [x,y,yaw] changes due to the control input.
    :param angle: The angle between x-axis and robot in rad
    :param deltak: The change in time from time step k-1 to k in sec
    """
    # parameters of the robot r = wheel radius, l = width of the robot
    r = 2.25
    l = 11

    B = np.array([[np.cos(angle) * deltak * r / 2, np.cos(angle) * deltak * r / 2],
                  [-np.sin(angle) * deltak * r / 2, -np.sin(angle) * deltak * r / 2],
                  [r / l * deltak, -r / l * deltak]])

    return B


def ekf(z_k_observation_vector, state_estimate_k_minus_1,
        control_vector_k_minus_1, P_k_minus_1, dk):
    """
    Extended Kalman Filter. Fuses noisy sensor measurement to
    create an optimal estimate of the state of the robotic system.

    INPUT
        :param z_k_observation_vector The observation from the Odometry
            3x1 NumPy Array [x,y,angle] in the global reference frame
            in [cm,cm,radians].
        :param state_estimate_k_minus_1 The state estimate at time k-1
            3x1 NumPy Array [x,y,angle] in the global reference frame
            in [cm,cm,radians].
        :param control_vector_k_minus_1 The control vector applied at time k-1
            2x1 NumPy Array [v_left,v_right] in the global reference frame
            in [cm per second,cm per second].
        :param P_k_minus_1 The state covariance matrix estimate at time k-1
            3x3 NumPy Array
        :param dk Time interval in seconds

    OUTPUT
        :return state_estimate_k near-optimal state estimate at time k
            3x1 NumPy Array ---> [cm,cm,radians]
        :return P_k state covariance_estimate for time k
            3x3 NumPy Array
    """
    ######################### Predict #############################
    # Predict the state estimate at k based on the state estimate at k-1 and the control input applied at k-1.
    # can @ be replaced by np.dot ?
    state_estimate_k = A_k_minus_1 @ (state_estimate_k_minus_1) + (getB(state_estimate_k_minus_1[2], dk)) @ (
        control_vector_k_minus_1) + (process_noise_v_k_minus_1)

    print(f'State Estimate Before EKF={state_estimate_k}')

    # Predict the state covariance estimate based on the previous
    # covariance and some noise
    P_k = A_k_minus_1 @ P_k_minus_1 @ A_k_minus_1.T + (Q_k)

    ################### Update ##########################
    # Calculate the difference between the actual sensor measurement at
    # time k minus what the measurement model predicted
    # the sensor measurements would be for the current timestep k.
    measurement_residual_y_k = z_k_observation_vector - ((H_k @ state_estimate_k) + (sensor_noise_w_k))

    print(f'Observation={z_k_observation_vector}')

    # Calculate the measurement residual covariance
    S_k = H_k @ P_k @ H_k.T + R_k

    # Calculate the near-optimal Kalman gain
    # We use pseudoinverse since some of the matrices might be
    # non-square or singular.
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)

    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)

    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)

    # Print the best (near-optimal) estimate of the current state of the robot
    print(f'State Estimate After EKF={state_estimate_k}')

    # Return the updated state and covariance estimates
    return state_estimate_k, P_k


def main():
    # We start at time k=1
    k = 1

    # Time interval in seconds
    dk = 1

    # Create a list of sensor observations at successive timesteps
    # Each list within z_k is an observation vector.
    z_k = np.array([[3.0, 3.1, 0.0],  # k=1
                    [3.1, 3.0, 0.41],  # k=2
                    [3.2, 3.1, 0.82],  # k=3
                    [3.3, 3.0, 1.23],  # k=4
                    [3.4, 3.1, 1.43]])  # k=5

    # The estimated state vector at time k-1 in the global reference frame.
    # [x_k_minus_1, y_k_minus_1, yaw_k_minus_1]
    # [cm, cm, radians]
    state_estimate_k_minus_1 = np.array([3.0, 3.0, 0.003])

    # The control input vector at time k-1 in the global reference frame.
    # [v_left, v_right]
    # [cm/second, cm/second]
    control_vector_k_minus_1 = np.array([1.0, -1.0])

    # State covariance matrix P_k_minus_1
    P_k_minus_1 = np.array([[0.1, 0, 0],
                            [0, 0.1, 0],
                            [0, 0, 0.1]])

    # Start at k=1 and go through each of the 5 sensor observations,
    # one at a time.
    # We stop right after timestep k=5 (i.e. the last sensor observation)
    for k, obs_vector_z_k in enumerate(z_k, start=1):
        # Print the current timestep
        print(f'Timestep k={k}')

        # Run the Extended Kalman Filter and store the
        # near-optimal state and covariance estimates
        optimal_state_estimate_k, covariance_estimate_k = ekf(
            obs_vector_z_k,  # Most recent sensor measurement
            state_estimate_k_minus_1,  # Our most recent estimate of the state
            control_vector_k_minus_1,  # Our most recent control input
            P_k_minus_1,  # Our most recent state covariance matrix
            dk)  # Time interval

        # Get ready for the next timestep by updating the variable values
        state_estimate_k_minus_1 = optimal_state_estimate_k
        P_k_minus_1 = covariance_estimate_k

        # Print a blank line
        print()


# Program starts running here with the main method
main()
