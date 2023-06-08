from flask import Flask, render_template, request, session, redirect, url_for, jsonify, Response, stream_template, stream_with_context, make_response
# from motor_vesc import VESC
import nidaqmx
import time
import serial
import numpy as np
import datetime
import pandas as pd
import os
import tkinter as tk
import json
import random
import serial
import logging
from queue import Queue
import threading
from threading import Thread
import pyvesc
from pandas import ExcelWriter
from pyvesc.VESC.messages import Alive, SetDutyCycle, SetRPM, GetValues


class ArduinoControl:
    def __init__(self, port):
        self.ser = serial.Serial(port=port, baudrate=9600, timeout=1)

        if not self.ser.is_open:
            self.ser.open()

        time.sleep(2)  # wait for Arduino to initialize

    def move_to(self, linear_position, rotary_position):
        # Send linear position command
        linear_position_str = '{}\n'.format(linear_position)
        command_state = 'linear'
        while True:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().strip().decode()
                if command_state == 'linear' and response == 'Enter linear position (mm):':
                    print('Arduino ready for linear position')
                    self.ser.write(linear_position_str.encode())
                    print('Arduino linear position sent')
                    command_state = 'linear-confirmation'
                elif response == 'OK':
                    print('Linear position set successfully')
                    # Now we'll send the rotary command
                    rotary_position_str = '{}\n'.format(rotary_position)
                    self.ser.write(rotary_position_str.encode())
                    command_state = 'rotary'
                elif command_state == 'rotary' and response.startswith('Position set'):
                    print(response)
                    break
                elif response.startswith('Error'):
                    raise ValueError(response)
                else:
                    print('Received:', response)

    def close(self):
        self.ser.close()


voltage_task = None
temperature_task = None
strain_task = None
experiment_running = False  # Flag to track if the experiment is runningt th

def configureDAQ(device_name, type, channels, sampling_rate, samples_per_channel, buffer_size=10000000):
    task = nidaqmx.Task()

    for channel in channels:
        full_channel_name = "{}/{}".format(device_name, channel)
        if type == 'voltage':
            try:
                task.ai_channels.add_ai_voltage_chan(full_channel_name, min_val=0, max_val=5)
                print("Added voltage channel:", full_channel_name)
            except nidaqmx.errors.DaqError as e:
                print("Error adding voltage channel:", full_channel_name)
                print("Error message:", str(e))
        elif type == 'strain':
            try:
                task.ai_channels.add_ai_force_bridge_two_point_lin_chan(full_channel_name,
                                                                        min_val=-100.0, max_val=100.0,
                                                                        units=nidaqmx.constants.ForceUnits.KILOGRAM_FORCE,
                                                                        bridge_config=nidaqmx.constants.BridgeConfiguration.FULL_BRIDGE,
                                                                        voltage_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL,
                                                                        voltage_excit_val=5, nominal_bridge_resistance=1000.0,
                                                                        electrical_units=nidaqmx.constants.BridgeElectricalUnits.MILLIVOLTS_PER_VOLT,
                                                                        physical_units=nidaqmx.constants.BridgePhysicalUnits.KILOGRAM_FORCE)
                print("Added strain channel:", full_channel_name)
            except nidaqmx.errors.DaqError as e:
                print("Error adding strain channel:", full_channel_name)
                print("Error message:", str(e))

    task.timing.cfg_samp_clk_timing(sampling_rate, samps_per_chan=samples_per_channel,
                                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
    task.in_stream.input_buf_size = buffer_size

    return task

def initializeDAQTasks(strain_device,
                       strain_channels,
                       strain_sampling_rate, strain_samples):
    
    global strain_task


    if strain_task is not None:
        strain_task.close()
        strain_task = None


    strain_task = configureDAQ(device_name=strain_device, type='strain', channels=strain_channels,
                               sampling_rate=strain_sampling_rate, samples_per_channel=strain_samples)

    tasks = {
        'strain': strain_task
    }

    return tasks


# initializeDAQTasks(
#     voltage_device='Voltage_DAQ',
#     strain_device='Strain_Device',
#     voltage_channels=['ai0', 'ai1', 'ai2', 'ai3'],
#     strain_channels=['ai0', 'ai1'],
#     voltage_sampling_rate=100,
#     voltage_samples=100,
#     strain_sampling_rate=100,
#     strain_samples=100
# )

# Read the data from the specified task
def readDAQData(task, samples_per_channel, channels, type):
    try:
        data = task.read(number_of_samples_per_channel=samples_per_channel)
        channel_data = {}

        if len(channels) == 1:
            channel_data[channels[0]] = data
        else:
            for i, channel in enumerate(channels):
                if type == 'voltage':
                    voltage_data = data[i]
                    pressure_data = []
                    for voltage in voltage_data:
                        output_percent = (voltage / 5.0) * 100.0
                        pressure = ((80.0 / 12.0) * (output_percent - 10.0)) - 6.0
                        pressure_data.append(pressure)
                    channel_data[channel] = pressure_data
                else:
                    channel_data[channel] = data[i]

        return channel_data

    except nidaqmx.errors.DaqReadError as e:
        print("Error while reading DAQ data:", e)
        return None

# This allows the app to start
app = Flask(__name__, static_url_path='/static', template_folder='templates')
app.secret_key = "secret_key"
# vesc = VESC("/dev/ttyACM0")
# motor_control = MotorControl(vesc)

export_csv_enabled = False  # Flag to track the export CSV button status

# Renders softwaremanual page when user clicks on the button
@app.route('/software_manual')
def software_manual():
    return render_template('softwaremanual.html')

# Renders inputparameters page when user clicks on the button
@app.route('/input_parameters')
def input_parameters():
    return render_template('inputparameters.html')

profiles = []

# Path to the JSON file to store the profiles
PROFILES_FILE = 'profiles.json'

@app.route('/save_profile', methods=['POST'])
def save_profile():
    profile_data = request.get_json()
    profiles.append(profile_data)

    # Write the updated profiles to the JSON file
    with open(PROFILES_FILE, 'w') as file:
        json.dump(profiles, file)

    # Update the session's input_motor_data with the new profile data
    session['input_motor_data'] = profile_data

    return jsonify(success=True)

@app.route('/get_profiles')
def get_profiles():
    with open(PROFILES_FILE, 'r') as file:
        profiles = json.load(file)
    return jsonify(profiles=profiles)

@app.route('/saved_profiles')
def saved_profiles():
    with open(PROFILES_FILE, 'r') as file:
        saved_profiles = json.load(file)

    return render_template('savedprofiles.html', profiles=saved_profiles)

@app.route('/get_profile_names', methods=['GET'])
def get_profile_names():
    # Retrieve the saved profile names from the profiles list
    profile_names = [profile['name'] for profile in profiles]

    return jsonify(profile_names)

@app.route('/delete_profile', methods=['DELETE'])
def delete_profile():
    profile_name = request.get_json().get('profileName')

    # Find and remove the profile with the matching name from the profiles list
    for profile in profiles:
        if profile['name'] == profile_name:
            profiles.remove(profile)
            break

    # Write the updated profiles list to the JSON file
    with open(PROFILES_FILE, 'w') as file:
        json.dump(profiles, file)

    return jsonify(success=True)


@app.route('/', methods=['GET'])
def index():
    input_motor_data = session.get('input_motor_data', {})
    last_values = session.get('last_values', {})
    time_data = session.get('time_data')
    json_p_zero_data = session.get('json_p_zero_data')
    json_p_one_data = session.get('json_p_one_data')
    json_p_two_data = session.get('json_p_two_data')
    json_p_three_data = session.get('json_p_three_data')
    json_strain_gauge_zero_data = session.get('json_strain_gauge_zero_data')
    json_strain_gauge_one_data = session.get('json_strain_gauge_one_data')
    json_motor_temp_data = session.get('json_motor_temp_data')

    return render_template('index.html', input_motor_data=input_motor_data, last_values=last_values, time_data=time_data, json_p_zero_data=json_p_zero_data, json_p_one_data=json_p_one_data, json_p_two_data=json_p_two_data, json_p_three_data=json_p_three_data, json_strain_gauge_zero_data=json_strain_gauge_zero_data, json_strain_gauge_one_data=json_strain_gauge_one_data, json_motor_temp_data=json_motor_temp_data, start_button_disabled=False)

# Collects all input parameters and saves it in a dictionary
@app.route('/motor_input_parameters', methods=['GET', 'POST'])
def motor_input_parameters():
    input_motor_data = {}

    error_duty_cycle_start = None
    error_duty_cycle_end = None
    error_duty_cycle_interval = None

    error_linear_actuator = None
    error_rotary_motor = None

    if request.method == 'POST':
        duty_cycle_start = int(request.form.get('duty_cycle_start'))
        if duty_cycle_start:
            try:
                duty_cycle_start = int(duty_cycle_start)
                if duty_cycle_start <0 or duty_cycle_start > 100:
                    error_duty_cycle_start = {'field': 'duty_cycle_start', 'message': 'Duty cycle must be between 0 and 100'}
                    duty_cycle_start = None
                    print(error_duty_cycle_start)
            except ValueError:
                error_duty_cycle_start = {'field': 'duty_cycle_start', 'message': 'Invalid input for duty cycle'}
                duty_cycle = None
                print(error_duty_cycle_start)
        else:
            error_duty_cycle_start = {'field': 'duty_cycle_start', 'message': 'Please enter a value for duty cycle'}
            duty_cycle_start = None
            print(error_duty_cycle_start)

        duty_cycle_end = int(request.form.get('duty_cycle_end'))
        if duty_cycle_end:
            try:
                duty_cycle_end = int(duty_cycle_end)
                if duty_cycle_end <0 or duty_cycle_end > 100:
                    error_duty_cycle_end = {'field': 'duty_cycle_end', 'message': 'Current must be between 0 and 5'}
                    current = None
                    print(error_duty_cycle_end)
            except ValueError:
                error_duty_cycle_end = {'field': 'duty_cycle_end', 'message': 'Invalid input for duty cycle'}
                duty_cycle_end = None
                print(error_duty_cycle_end)
        else:
            error_duty_cycle_end = {'field': 'duty_cycle_end', 'message': 'Please enter a value for duty cycle'}
            duty_cycle_end = None
            print(error_duty_cycle_end)

        duty_cycle_interval = float(request.form.get('duty_cycle_interval'))
        if duty_cycle_interval:
            try:
                duty_cycle_interval = int(duty_cycle_interval)
                if duty_cycle_interval <0 or duty_cycle_interval > 100:
                    error_duty_cycle_interval = {'field': 'duty_cycle_interval', 'message': 'Duty Cycle Interval must be between 0 and 100'}
                    current = None
                    print(error_duty_cycle_interval)
            except ValueError:
                error_duty_cycle_interval = {'field': 'duty_cycle_interval', 'message': 'Invalid input for duty cycle interval'}
                duty_cycle_interval = None
                print(error_duty_cycle_interval)
        else:
            error_duty_cycle_interval = {'field': 'duty_cycle_interval', 'message': 'Please enter a value for duty cycle'}
            duty_cycle_interval = None
            print(error_duty_cycle_interval)


        linear_actuator = request.form.get('linear_actuator')
        if linear_actuator:
            try:
                linear_actuator = float(linear_actuator)
                if linear_actuator < 0 or linear_actuator > 1000:
                    error_linear_actuator = {'field': 'linear_actuator', 'message': 'Actuator position must be between 0 and 100'}
                    linear_actuator = None
                    print(error_linear_actuator)
            except ValueError:
                error_linear_actuator = {'field': 'linear_actuator', 'message': 'Invalid input for actuator position'}
                print(error_linear_actuator)
        else:
            error_linear_actuator = {'field': 'linear_actuator', 'message': 'Please enter a value for actuator position'}
            linear_actuator = None
            print(error_linear_actuator)
        
        rotary_motor = request.form.get('rotary_motor')
        if rotary_motor:
            try:
                rotary_motor = float(rotary_motor)
                if rotary_motor < 0 or rotary_motor > 360:
                    error_rotary_motor = {'field': 'rotary_motor', 'message': 'Rotary motor position must be between 0 and 360'}
                    rotary_motor = None
                    print(error_rotary_motor)
            except ValueError:
                error_rotary_motor = {'field': 'rotary_motor', 'message': 'Invalid input for rotary position'}
                print(error_rotary_motor)
        else:
            error_rotary_motor = {'field': 'rotary_motor', 'message': 'Please enter a value for rotary position'}
            rotary_motor = None
            print(error_rotary_motor)

        if error_duty_cycle_start or error_duty_cycle_end or error_duty_cycle_interval or error_linear_actuator or error_rotary_motor:
            session['input_motor_data'] = None
            return render_template('inputparameters.html', error_duty_cycle_start=error_duty_cycle_start, error_duty_cycle_end=error_duty_cycle_end, error_duty_cycle_interval = error_duty_cycle_interval, error_linear_actuator=error_linear_actuator, error_rotary_motor=error_rotary_motor, input_motor_data=input_motor_data)

        input_motor_data['duty_cycle_start'] = duty_cycle_start
        input_motor_data['duty_cycle_end'] = duty_cycle_start
        input_motor_data['duty_cycle_interval'] = duty_cycle_interval
        input_motor_data['linear_actuator'] = linear_actuator
        input_motor_data['rotary_motor'] = rotary_motor
        input_motor_data['vesc_port'] = request.form.get('vesc_port')
        input_motor_data['arduino_port'] = request.form.get('arduino_port')
        session['input_motor_data'] = input_motor_data

        print(input_motor_data)

    input_motor_data = session.get('input_motor_data', {})
    return render_template('inputparameters.html', input_motor_data=input_motor_data)

# Clears the input parameters from the session
@app.route('/reset_session', methods=['POST'])
def reset_session():
    session.pop('input_motor_data', None)
    return redirect(url_for('index'))

@app.route('/stop_button', methods=['POST'])
def stop_button():
    return render_template('index.html')

logging.basicConfig(level=logging.DEBUG)
experiment_running = False # Initialise experiment_running
data_lock = threading.Lock() # Initialise the data lock
stop_event = threading.Event() # Initialise the stop event

# Initialise values
pdiff1_recent = 1.0
pdiff2_recent = 1.0
pdiff3_recent = 1.0
pdiff_average = 0.0
k_beta = 0.0
k_t = 0.0
velocity = 0.0
yaw_angle = 0.0

strain1_recent = 0.0
strain2_recent = 0.0
motor_duty_cycle_recent = 0.0
# motor_rpm_recent = 0.0
ser = None  # Declare ser globally
motor_temp_recent = 0.0
pressure_data_df = pd.DataFrame(columns = ['time', 'pdiff1', 'pdiff2', 'pdiff3', 'k_beta', 'k_t', 'yaw_angle', 'velocity']) # Initialise the data dataframe
strain_data_df = pd.DataFrame(columns = ['time', 'Thrust 1', 'Thrust 2']) # Initialise the data dataframe

pdiff_queue = Queue() # Initialise the queue for the pdiff values
strain_queue = Queue() # Initialise the queue for the strain values
motor_queue= Queue() # Initialise the queue for the motor values

#start a timer to keep track of the experiment duration
start_time = time.time()

def read_pdiff_values():
    pressure_serial = serial.Serial('COM6', 9600)  # Replace 'COM6' with the appropriate serial port
    global experiment_running, pdiff1_recent, pdiff2_recent, pdiff3_recent, velocity, yaw_angle

    while experiment_running:
        line = pressure_serial.readline().decode().strip()  # Read a line from the serial port and decode it
        if line:
            values = line.split(',')  # Split the line by comma to extract the pdiff values
            
            if len(values) >= 3:
                try:
                    pdiff1_recent = float(values[0])  # Convert the first value to float
                    pdiff2_recent = float(values[1])  # Convert the second value to float
                    pdiff3_recent = float(values[2])  # Convert the third value to float

                    density = 1.392181 # kg/m^3

                    pdiff_average = ((pdiff1_recent + pdiff2_recent + pdiff3_recent) / 3)
                    k_beta = (pdiff2_recent - pdiff3_recent) / (pdiff1_recent - pdiff_average)
                    k_t = -0.01617818*k_beta**8 + 0.021501133*k_beta**7 + 0.06570708*k_beta**6 - 0.008913*k_beta**5 - 0.27974366*k_beta**4 + 0.06411619*k_beta**3 + 0.138739*k_beta**2 - 0.00876*k_beta + 0.005485
                    velocity = abs(((2*(pdiff1_recent - (k_t * (pdiff1_recent-pdiff_average))))/density))**0.5
                    yaw_angle = 0.039989809*k_beta**8 - 0.159177308*k_beta**7 - 0.2904159*k_beta**6 - 0.1602285*k_beta**5 - 0.3923435*k_beta**4 + 0.29777905*k_beta**3 + 1.917721*k_beta**2 - 18.7517*k_beta - 0.51817
                    
                    elapsed_time = time.time() - start_time
                    pressure_data_df = pd.concat[pressure_data_df, pd.DataFrame([[elapsed_time, pdiff1_recent, pdiff2_recent, pdiff3_recent, k_beta, k_t, yaw_angle, velocity]], columns = ['time', 'pdiff1', 'pdiff2', 'pdiff3', 'k_beta', 'k_t', 'yaw_angle', 'velocity'])] # Append the sample dataframe to the data dataframe

                    pdiff_queue.put([pdiff1_recent, pdiff2_recent, pdiff3_recent, velocity, yaw_angle]) # Put the values in the queue         
                
                except ZeroDivisionError:
                    velocity = 0.0
                    pdiff1_recent = 1.0
                    pdiff2_recent = 1.0
                    pdiff3_recent = 1.0
                    yaw_angle = 0.0

                    pressure_data_df = pd.concat([pressure_data_df, pd.DataFrame([[pdiff1_recent, pdiff2_recent, pdiff3_recent, k_beta, k_t, yaw_angle, velocity]], columns = ['pdiff1', 'pdiff2', 'pdiff3', 'k_beta', 'k_t', 'yaw_angle', 'velocity'])])

                    pdiff_queue.put([pdiff1_recent, pdiff2_recent, pdiff3_recent, velocity, yaw_angle]) # Put the values in the queue

    pressure_serial.close()
    return pdiff1_recent, pdiff2_recent, pdiff3_recent, velocity, yaw_angle


def read_strain_values():
    global experiment_running
    global sample_df

    global strain_device
    global strain_channels
    global strain_sampling_rate
    global strain_samples

     # Retrieve the strain gauge offsets from the session
    # strain_gauge_offset_1 = session.get('strain_gauge_offset_1')
    # strain_gauge_offset_2 = session.get('strain_gauge_offset_2')

    # logging.DEBUG(experiment_running)


    # Check if the experiment is not running
    if experiment_running == False:
        return # Exit the function if the experiment is not running
    

    # Define the channels and parameters for each type of data
    strain_device = 'Strain_Device'
    strain_channels = ['ai0', 'ai1']
    strain_sampling_rate = 10
    strain_samples = 3

    # Create empty pandas dataframe to store data

    # Initialize the DAQ tasks
    tasks = initializeDAQTasks(strain_device=strain_device,
                               strain_channels=strain_channels,
                               strain_sampling_rate=strain_sampling_rate,
                               strain_samples=strain_samples)

    strain_task = tasks['strain']


    while experiment_running:
        strain_data = readDAQData(strain_task, samples_per_channel=strain_samples, channels=strain_channels,
                                type='strain')
        

        if strain_data is not None:
            # Add the data to the DataFrame
            current_time = datetime.datetime.now()
            num_samples = len(strain_data[strain_channels[0]])
            seconds_per_sample = 1.0 / strain_sampling_rate
            seconds = np.arange(num_samples) * seconds_per_sample

            sample = {'Time': [current_time] * num_samples, 'Seconds': seconds}


            for i, channel in enumerate(strain_channels):
                column_name = 'Strain Measurement {}'.format(i)
                sample[column_name] = pd.Series(strain_data[channel])

            # Convert the sample dictionary to a DataFrame
            sample_df = pd.DataFrame(sample)

            # # Apply offsets to each strain measurement column
            # if strain_gauge_offset_1 is not None:
            #     sample_df['Strain Measurement 0'] = sample_df['Strain Measurement 0'].apply(lambda x: -1 * (x + strain_gauge_offset_1))
            # if strain_gauge_offset_2 is not None:
            #     sample_df['Strain Measurement 1'] = sample_df['Strain Measurement 1'].apply(lambda x: x + strain_gauge_offset_2)
            


        strain_gauge_zero_data = sample_df[['Seconds', 'Strain Measurement 0']]
        strain_gauge_one_data = sample_df[['Seconds', 'Strain Measurement 1']]

        strain1_recent = strain_gauge_zero_data['Strain Measurement 0'].iloc[-1]
        strain2_recent = strain_gauge_one_data['Strain Measurement 1'].iloc[-1]

        elapsed_time = time.time() - start_time
        strain_data_df = pd.concat([strain_data_df, pd.DataFrame([[elapsed_time, strain1_recent, strain2_recent]], columns = ['time', 'Thrust 1', 'Thrust 2'])])

        strain_queue.put([strain1_recent, strain2_recent]) # Put the values in the queue

        # # Store the required dataframes in the session
        # session['time_data'] = time_data
        # session['json_strain_gauge_zero_data'] = json_strain_gauge_zero_data
        # session['json_strain_gauge_one_data'] = json_strain_gauge_one_data

    return strain1_recent, strain2_recent

def read_motor_values(ser):
    global experiment_running, stop_event, motor_duty_cycle_recent, motor_temp_recent

    while experiment_running and not stop_event.is_set():
        # Check if there is enough data back for a measurement
        if ser.in_waiting > 71:
            (response, consumed) = pyvesc.decode(ser.read(ser.in_waiting))
            # Decode and process the response
            try:
                if response:
                    motor_duty_cycle_recent = response.duty_cycle_now
                    motor_temp_recent = response.temp_fet
                    motor_queue.put([motor_duty_cycle_recent, motor_temp_recent])
            except Exception as e:
                print(f"Error processing response: {str(e)}")

def start_reading_motor_values():
    global experiment_running, stop_event
    # Start the data reading thread
    motor_queue.queue.clear()
    logging.debug('Starting motor thread')
    thread_motor = threading.Thread(target=start_motor, args=(ser,))
    thread_motor_values = threading.Thread(target=read_motor_values, args=(ser,))
    logging.debug('Starting motor values thread')
    thread_motor.start()
    thread_motor_values.start()


def start_reading_pdiff_values():
    global experiment_running, stop_event
    # Start the data reading thread (threading allows multiple tasks to run simultaneously)
    pdiff_queue.queue.clear()
    strain_queue.queue.clear()
    stop_event.clear()
    thread_pdiff = threading.Thread(target=read_pdiff_values)
    thread_pdiff.start()

def start_reading_strain_values():
    global experiment_running, stop_event
    pdiff_queue.queue.clear()
    strain_queue.queue.clear()
    stop_event.clear()
    thread_strain = threading.Thread(target=read_strain_values)
    thread_strain.start()

# def start_reading_motor_values():
#     global experiment_running, stop_event
#     stop_event.clear()
#     thread_motor = threading.Thread(target=read_motor_values)
#     thread_motor.start()
    
# This removes the delay between the front and backend by ensuring the pdiff values is synchronised between threads
def get_recent_pdiff_values():
    global pdiff1_recent, pdiff2_recent, pdiff3_recent, velocity, yaw_angle
    with data_lock:
        return pdiff1_recent, pdiff2_recent, pdiff3_recent, velocity, yaw_angle
    

def get_recent_strain_values():
    global strain1_recent, strain2_recent
    with data_lock:
        return strain1_recent, strain2_recent
    
def get_recent_motor_values():
    global motor_duty_cycle_recent, motor_temp_recent
    with data_lock:
        return motor_duty_cycle_recent, motor_temp_recent


@app.route('/start_experiment', methods=['GET', 'POST'])
def start_experiment():
    global experiment_running, ser
    logging.debug('Starting experiment.')
    if not experiment_running:
        experiment_running = True

        start_reading_pdiff_values()
        start_reading_strain_values()
        # start_reading_motor_values()

    return "Started"

@app.route('/stop_experiment')
def stop_experiment():
    global experiment_running, ser
    experiment_running = False
    stop_event.set()
    if ser is not None:
        ser.close()  # Close the serial connection
        ser = None
    return 'Experiment stopped'


@app.route('/pdiff_data')
def pdiff_data():
    global pdiff_queue
    # Checks if the queue is empty
    if not pdiff_queue.empty():
        # Retrieves most recent values from the queue
        pdiff = pdiff_queue.get()
        # flowvelocity = velocity_queue.get()
    else:
        pdiff = get_recent_pdiff_values()
        # flowvelocity = get_recent_pdiff_values()
    return jsonify(pdiff)

@app.route('/strain_data')
def strain_data():
    global strain_queue
    if not strain_queue.empty():
        strain = strain_queue.get()
    else:
        strain = get_recent_strain_values()
    return jsonify(strain)

@app.route('/calibrate_load_cells', methods=['POST'])
def calibrate_load_cells():

    global strain_device
    global strain_channels
    global strain_sampling_rate
    global strain_samples

    strain_channels = ['ai0', 'ai1']

    strain_task = configureDAQ(device_name='Strain_Device', type='strain', channels=strain_channels,
                               sampling_rate=1000, samples_per_channel=300)
    
    strain_data = readDAQData(strain_task, samples_per_channel=300, channels=["ai0", "ai1"], type='strain')

 # Calculate the average value for each load cell
    averages = []
    for channel in strain_channels:
        if channel in strain_data:
            values = strain_data[channel]
            average = np.mean(values)
            averages.append(average)

    # Close the DAQ task
    strain_task.close()

    # Store the offsets in the session
    session['strain_gauge_offset_1'] = averages[0]
    session['strain_gauge_offset_2'] = averages[1]

    # Return the calibrated offsets as JSON
    return jsonify({
        'offsets': {
            'strain_gauge_offset_1': averages[0],
            'strain_gauge_offset_2': averages[1]
        }
    })


arduino = None

def establish_arduino_connection():
    global arduino

    if arduino is None:
        try:
            input_motor_data = session.get('input_motor_data', {})
            arduino = ArduinoControl(input_motor_data['arduino_port'])
        except Exception as e:
            return "Error: Failed to establish Arduino connection.", 400
        else:
            print(f"establish_arduino_connection: arduino is {arduino}")

def move_linear_and_rotary_actuator(linear_position, rotary_position):
    
    global arduino

    try:
        # Move the linear and rotary actuators to the specified positions
        arduino.move_to(linear_position, rotary_position)

    except ValueError as e:
        print('Error:', str(e))



@app.route('/start_all', methods=['POST'])
def start_all():

    global experiment_running, ser
    input_motor_data = session.get('input_motor_data', {})
    # Set the experiment_running flag to True
    experiment_running = True

    # Initialize the last_values dictionary
    last_values = {}


    #establish_arduino_connection()  # Assign the returned arduino object
    #start_actuators()

    #start_motor(ser)

    

    return redirect(url_for('index'))

# @app.route('/start_all', methods=['POST'])
# def start_all():
#     global experiment_running

#     # Check if the experiment is already running
#     if experiment_running == False:
#         input_motor_data = session.get('input_motor_data', {})
#         global vesc
#         global arduino

#         # Set the experiment_running flag to True
#         experiment_running = True

#         # Disable the start button
#         session['start_button_disabled'] = True
        
#         # Initialize the last_values dictionary
#         while experiment_running == True:

#             # Call the main function to start the data acquisition and get the updated last_values

#             time_data, json_p_zero_data, json_p_one_data, json_p_two_data, json_p_three_data, json_strain_gauge_zero_data, json_strain_gauge_one_data, json_motor_temp_data = main()

#             # Store the last values in the session
#             session['time_data'] = time_data
#             session['json_p_zero_data'] = json_p_zero_data
#             session['json_p_one_data'] = json_p_one_data
#             session['json_p_two_data'] = json_p_two_data
#             session['json_p_three_data'] = json_p_three_data
#             session['json_strain_gauge_zero_data'] = json_strain_gauge_zero_data
#             session['json_strain_gauge_one_data'] = json_strain_gauge_one_data
#             session['json_motor_temp_data'] = json_motor_temp_data

#             # Render the template with updated values
#             return render_template('index.html', input_motor_data=input_motor_data, last_values=last_values, time_data=time_data, json_p_zero_data=json_p_zero_data, json_p_one_data=json_p_one_data, json_p_two_data=json_p_two_data, json_p_three_data=json_p_three_data, json_strain_gauge_zero_data=json_strain_gauge_zero_data, json_strain_gauge_one_data=json_strain_gauge_one_data, json_motor_temp_data=json_motor_temp_data, start_button_disabled=session.get('start_button_disabled', False))

    return redirect(url_for('index'))


def start_motor(ser):
    
    global experiment_running, stop_event
    logging.debug("start_motor: start_motor called")
    try:
        i = 5
        while experiment_running and not stop_event.is_set():
            for j in range(5 * 100):
                ser.write(pyvesc.encode(Alive()))  # Send heartbeat
                ser.write(pyvesc.encode(SetDutyCycle(i / 100)))  # Send command to get values
                ser.write(pyvesc.encode_request(GetValues))
                time.sleep(0.01)
            if i < 15:
                i += 1
    except Exception as e:
        print(f"Error: {str(e)}")
  




def start_actuators():

    # Retrieve linear actuator and rotary motor positions from session
    input_motor_data = session.get('input_motor_data', {})
    linear_position = input_motor_data.get('linear_actuator', 0)
    rotary_position = input_motor_data.get('rotary_motor', 0)


    # Move the linear actuator to the specified positions
    print(f"start_actuators: Before move_linear_and_rotary_actuator, arduino is {arduino}")
    try:
        move_linear_and_rotary_actuator(linear_position, rotary_position)
    except ValueError as e:
        return str(e), 400
    print(f"start_actuators: After move_linear_and_rotary_actuator, arduino is {arduino}")



data_df = pd.DataFrame()  # Create an empty dataframe to store the collected data

# def update_chart():
#     global data_df  # Access the global dataframe

#     # Retrieve the data from the DAQ
#     json_p_zero_data, json_p_one_data, json_p_two_data, json_p_three_data, json_strain_gauge_one_data, json_strain_gauge_two_data = main()

#     # Convert the JSON data to dataframes
#     p_zero_data = pd.read_json(json_p_zero_data)
#     p_one_data = pd.read_json(json_p_one_data)
#     p_two_data = pd.read_json(json_p_two_data)
#     p_three_data = pd.read_json(json_p_three_data)
#     strain_gauge_one_data = pd.read_json(json_strain_gauge_one_data)
#     strain_gauge_two_data = pd.read_json(json_strain_gauge_two_data)

#     # Concatenate the new data horizontally
#     new_data = pd.concat([p_zero_data, p_one_data, p_two_data, p_three_data,
#                                 strain_gauge_one_data, strain_gauge_two_data], axis=1)

#     # Concatenate the horizontally concatenated dataframe with data_df vertically
#     data_df = pd.concat([data_df, new_data], axis=0, ignore_index=True)

#     # Render the index.html template and pass the JSON data and the dataframe to the template
#     return render_template('index.html',
#                            json_p_zero_data=json_p_zero_data,
#                            json_p_one_data=json_p_one_data,
#                            json_p_two_data=json_p_two_data,
#                            json_p_three_data=json_p_three_data,
#                            json_strain_gauge_one_data=json_strain_gauge_one_data,
#                            json_strain_gauge_two_data=json_strain_gauge_two_data,
#                            data_df=data_df)

@app.route('/update_values', methods=['GET'])
def update_values():

    global last_values
    last_values = session.get('last_values', {})
    
    # Perform any necessary processing on last_values here
    
    # Generate the HTML for last values
    last_values_html = ''
    for key, value in last_values.items():
        last_values_html += f'<p>{key}: {value}</p>'

    print('Updated values from update_values() {}'.format(last_values))

    return {'last_values_html': last_values_html}


@app.route('/export_csv', methods=['POST'])
def export_csv():
    global pressure_data_df, strain_data_df, export_csv_enabled

    if export_csv_enabled:
        # Define the file path and name on the server
        file_path = 'data.xlsx'  # Change the extension to .xlsx

        # Export the dataframes to the file path
        with ExcelWriter(file_path) as writer:
            pressure_data_df.to_excel(writer, sheet_name='Pressure', index=False)
            strain_data_df.to_excel(writer, sheet_name='Strain', index=False)

        # Check if the file was successfully saved
        if os.path.isfile(file_path):
            return "CSV Exported: " + file_path
        else:
            return "CSV Export Failed"
    else:
        return "CSV Export is not enabled"

@app.route('/stop', methods=['GET', 'POST'])
def stop():
    global experiment_running
    global arduino

    # Check if the experiment is not running
    if not experiment_running:
        return redirect(url_for('index'))  # Redirect to the main page
    
    # save the data to a csv file
    # save_data_to_csv()

    # Update the export CSV button status
    export_csv_enabled = True

    # Stop the actuators
    #stop_actuators()

    # Set the experiment_running flag to False
    experiment_running = False

    #stop_motor()

    return redirect(url_for('index'))  # Redirect to the main page


def stop_motor():
    global ser
    ser.write(pyvesc.encode(SetDutyCycle(0)))  # Send command to set duty cycle to zero
    ser.close()  # Close the serial connection

def stop_actuators():

    global arduino

    # Retrieve linear actuator and rotary motor positions from session
    input_motor_data = session.get('input_motor_data', {})

    try:
        arduino = ArduinoControl(input_motor_data['arduino_port'])
    except:
        return "Error: Arduino port connection not found.", 400
    
    # Move the actuators back to the 0 position
    arduino.move_to(0, 0.0000001)  # Adjust the values accordingly if needed

    return "Actuators stopped successfully!"


if __name__ == '__main__':
    app.run(debug=True) 
