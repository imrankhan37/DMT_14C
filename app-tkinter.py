
from flask import Flask, render_template, request, session, redirect, url_for, jsonify, Response, stream_template, stream_with_context, make_response
import threading
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
import threading
import logging
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

voltage_task = None
temperature_task = None
strain_task = None
experiment_running = False  # Flag to track if the experiment is running

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

def initializeDAQTasks(voltage_device, strain_device,
                       voltage_channels, strain_channels,
                       voltage_sampling_rate, voltage_samples,
                       strain_sampling_rate, strain_samples):
    
    global voltage_task
    global temperature_task
    global strain_task

    if voltage_task is not None:
        voltage_task.close()
        voltage_task = None


    if strain_task is not None:
        strain_task.close()
        strain_task = None

    voltage_task = configureDAQ(device_name=voltage_device, type='voltage', channels=voltage_channels,
                                sampling_rate=voltage_sampling_rate, samples_per_channel=voltage_samples)

    strain_task = configureDAQ(device_name=strain_device, type='strain', channels=strain_channels,
                               sampling_rate=strain_sampling_rate, samples_per_channel=strain_samples)

    tasks = {
        'voltage': voltage_task,
        'strain': strain_task
    }

    return tasks


initializeDAQTasks(
    voltage_device='Voltage_DAQ',
    strain_device='Strain_Device',
    voltage_channels=['ai0', 'ai1', 'ai2', 'ai3'],
    strain_channels=['ai0', 'ai1'],
    voltage_sampling_rate=100,
    voltage_samples=100,
    strain_sampling_rate=100,
    strain_samples=100
)


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
    # Read the profiles from the JSON file
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

    error_duty_cycle = None
    error_current = None
    error_speed = None
    error_ramp_up_speed = None
    error_ramp_down_speed = None

    error_linear_actuator = None
    error_rotary_motor = None

    if request.method == 'POST':
        duty_cycle = request.form.get('duty_cycle')
        if duty_cycle:
            try:
                duty_cycle = float(duty_cycle)
                if duty_cycle <0 or duty_cycle > 100:
                    error_duty_cycle = {'field': 'duty_cycle', 'message': 'Duty cycle must be between 0 and 100'}
                    duty_cycle = None
                    print(error_duty_cycle)
            except ValueError:
                error_duty_cycle = {'field': 'duty_cycle', 'message': 'Invalid input for duty cycle'}
                duty_cycle = None
                print(error_duty_cycle)
        else:
            error_duty_cycle = {'field': 'duty_cycle', 'message': 'Please enter a value for duty cycle'}
            duty_cycle = None
            print(error_duty_cycle)

        current = request.form.get('current')
        if current:
            try:
                current = float(current)
                if current <0 or current > 5:
                    error_current = {'field': 'current', 'message': 'Current must be between 0 and 5'}
                    current = None
                    print(error_current)
            except ValueError:
                error_current = {'field': 'current', 'message': 'Invalid input for current'}
                current = None
                print(error_current)
        else:
            error_current = {'field': 'current', 'message': 'Please enter a value for current'}
            current = None
            print(error_current)

        speed = request.form.get('speed')
        if speed:
            try:
                speed = float(speed)
                if speed < 0 or speed > 10000:
                    error_speed = {'field': 'speed', 'message': 'Speed must be between 0 and 10,000'}
                    speed = None
                    print(error_speed)
            except ValueError:
                error_speed = {'field': 'speed', 'message': 'Invalid input for speed'}
                print(error_speed)
        else:
            error_speed = {'field': 'speed', 'message': 'Please enter a value for speed'}
            speed = None
            print(error_speed)
        
        ramp_down_speed = request.form.get('ramp_down_speed')
        if ramp_down_speed:
            try:
                ramp_down_speed = float(ramp_down_speed)
                if ramp_down_speed >= speed or ramp_down_speed > 10000:
                    error_ramp_down_speed = {'field': 'ramp_down_speed', 'message': 'Ramp down speed must be between 0 and 10,000 and less than input speed'}
                    ramp_down_speed = None
                    print(error_ramp_down_speed)
            except ValueError:
                error_ramp_down_speed = {'field': 'ramp_down_speed', 'message': 'Invalid input for ramp down speed'}
                print(error_ramp_down_speed)
        else:
            error_ramp_down_speed = {'field': 'ramp_down_speed', 'message': 'Please enter a value for ramp down speed'}
            ramp_down_speed = None
            print(error_ramp_down_speed)
           
        ramp_up_speed = request.form.get('ramp_up_speed')
        if ramp_up_speed:
            try:
                ramp_up_speed = float(ramp_up_speed)
                if ramp_up_speed <= speed or ramp_up_speed > 10000:
                    error_ramp_up_speed = {'field': 'ramp_up_speed', 'message': 'Ramp up speed must be between 0 and 10,000 and greater than input speed'}
                    ramp_up_speed = None
                    print(error_ramp_up_speed)
            except ValueError:
                error_ramp_up_speed = {'field': 'ramp_up_speed', 'message': 'Invalid input for ramp up speed'}
                print(error_ramp_up_speed)
        else:
            error_ramp_up_speed = {'field': 'ramp_up_speed', 'message': 'Please enter a value for ramp up speed'}
            ramp_up_speed = None
            print(error_ramp_up_speed)

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

        if error_duty_cycle or error_current or error_speed or error_ramp_down_speed or error_ramp_up_speed or error_linear_actuator or error_rotary_motor:
            session['input_motor_data'] = None
            return render_template('inputparameters.html', error_duty_cycle=error_duty_cycle, error_current=error_current, error_speed = error_speed, error_ramp_down_speed=error_ramp_down_speed, error_ramp_up_speed=error_ramp_up_speed, error_linear_actuator=error_linear_actuator, error_rotary_motor=error_rotary_motor, input_motor_data=input_motor_data)

        input_motor_data['duty_cycle'] = duty_cycle
        input_motor_data['current'] = current
        input_motor_data['speed'] = speed
        input_motor_data['ramp_down_speed'] = ramp_down_speed
        input_motor_data['ramp_up_speed'] = ramp_up_speed
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

experiment_running = True


@app.route('/main', methods=['POST'])
def main():
    global last_values
    global experiment_running
    global sample_df

    global strain_device
    global strain_channels
    global strain_sampling_rate
    global strain_samples

     # Retrieve the strain gauge offsets from the session
    strain_gauge_offset_1 = session.get('strain_gauge_offset_1')
    strain_gauge_offset_2 = session.get('strain_gauge_offset_2')

    # Check if the experiment is not running
    if experiment_running == False:
        return # Exit the function if the experiment is not running

    # Define the channels and parameters for each type of data
    voltage_device = 'Voltage_DAQ'
    strain_device = 'Strain_Device'
    voltage_channels = ['ai1', 'ai2', 'ai3', 'ai4']
    strain_channels = ['ai1', 'ai2']
    voltage_sampling_rate = 300
    voltage_samples = 20

    strain_sampling_rate = 300
    strain_samples = 20

    # Create empty pandas dataframe to store data
    data_df = pd.DataFrame(columns=['Voltage Measurement {}'.format(i) for i in range(len(voltage_channels))] +
                                 ['Strain Measurement {}'.format(i) for i in range(len(strain_channels))])

    # Initialize the DAQ tasks
    tasks = initializeDAQTasks(voltage_device=voltage_device,
                               strain_device=strain_device,
                               voltage_channels=voltage_channels,
                               strain_channels=strain_channels,
                               voltage_sampling_rate=voltage_sampling_rate,
                               voltage_samples=voltage_samples,
                               strain_sampling_rate=strain_sampling_rate,
                               strain_samples=strain_samples)

    voltage_task = tasks['voltage']
    strain_task = tasks['strain']

    time_data = '[]'


    while True:
        try:
            # Read the data from the DAQ tasks and update last_values accordingly
            voltage_data = readDAQData(voltage_task, samples_per_channel=voltage_samples, channels=voltage_channels,
                                    type='voltage')
            strain_data = readDAQData(strain_task, samples_per_channel=strain_samples, channels=strain_channels,
                                    type='strain')

            if voltage_data is not None  and strain_data is not None:
                # Add the data to the DataFrame
                current_time = datetime.datetime.now()
                num_samples = len(voltage_data[voltage_channels[0]])
                seconds_per_sample = 1.0 / voltage_sampling_rate
                seconds = np.arange(num_samples) * seconds_per_sample

                sample = {'Time': [current_time] * num_samples, 'Seconds': seconds}

                for i, channel in enumerate(voltage_channels):
                    column_name = 'Voltage Measurement {}'.format(i)
                    sample[column_name] = pd.Series(voltage_data[channel])

                for i, channel in enumerate(strain_channels):
                    column_name = 'Strain Measurement {}'.format(i)
                    sample[column_name] = pd.Series(strain_data[channel])

                # Convert the sample dictionary to a DataFrame
                sample_df = pd.DataFrame(sample)

                # Apply offsets to each strain measurement column
                if strain_gauge_offset_1 is not None:
                    sample_df['Strain Measurement 0'] = sample_df['Strain Measurement 0'].apply(lambda x: -1 * (x + strain_gauge_offset_1))
                if strain_gauge_offset_2 is not None:
                    sample_df['Strain Measurement 1'] = sample_df['Strain Measurement 1'].apply(lambda x: x + strain_gauge_offset_2)
                
                # Append the sample dataframe to the data dataframe
                print(sample_df)

            # Append the sample dataframe to the data dataframe
            data_df = pd.concat([data_df, sample_df], ignore_index=True)

            # Update the last values dictionary
            p_zero_data = sample_df[['Seconds', 'Voltage Measurement 0']]
            p_one_data = sample_df[['Seconds', 'Voltage Measurement 1']]
            p_two_data = sample_df[['Seconds', 'Voltage Measurement 2']]
            p_three_data = sample_df[['Seconds', 'Voltage Measurement 3']]
            strain_gauge_zero_data = sample_df[['Seconds', 'Strain Measurement 0']]
            strain_gauge_one_data = sample_df[['Seconds', 'Strain Measurement 1']]


        except Exception as e:
            print("An error occurred:", str(e))
            voltage_task.close()
            strain_task.close()

        voltage_task.close()
        strain_task.close()

        return time_data, p_zero_data, p_one_data, p_two_data, p_three_data, strain_gauge_zero_data, strain_gauge_one_data



@app.route('/calibrate_load_cells', methods=['POST'])
def calibrate_load_cells():

    global strain_device
    global strain_channels
    global strain_sampling_rate
    global strain_samples

    strain_channels = ['ai0', 'ai1']

    strain_task = configureDAQ(device_name='Strain_Device', type='strain', channels=strain_channels,
                               sampling_rate=1000, samples_per_channel=300)
    
    data = readDAQData(strain_task, samples_per_channel=300, channels=["ai0", "ai1"], type='strain')

 # Calculate the average value for each load cell
    averages = []
    for channel in strain_channels:
        if channel in data:
            values = data[channel]
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



@app.route('/start_all', methods=['POST'])
def start_all():

    global experiment_running

    # Set the experiment_running flag to True
    experiment_running = True

    root = tk.Tk()
    root.title("Data Visualization")
    root.geometry("800x400")

    # Create the matplotlib figure and axes
    fig = Figure(figsize=(6, 4), dpi=100)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    try:
        while experiment_running:
            # Create the Tkinter window

            time_data, p_zero_data, p_one_data, p_two_data, p_three_data, strain_gauge_zero_data, strain_gauge_one_data = main()

             # Function to update the plots with new data
            def update_plots(time_data, p_zero_data, p_one_data, p_two_data, p_three_data, strain_gauge_zero_data, strain_gauge_one_data):
                # Clear the axes
                ax1.clear()
                ax2.clear()

                # Plot the pressure data against time
                ax1.plot(time_data, p_zero_data, label='P0')
                ax1.plot(time_data, p_one_data, label='P1')
                ax1.plot(time_data, p_two_data, label='P2')
                ax1.plot(time_data, p_three_data, label='P3')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Pressure')
                ax1.legend()

                # Plot the strain gauge data against time
                ax2.plot(time_data, strain_gauge_zero_data, label='Strain 0')
                ax2.plot(time_data, strain_gauge_one_data, label='Strain 1')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Strain')
                ax2.legend()

                # Update the canvas
                canvas.draw()

            # Call the update_plots function with the data
            update_plots(time_data, p_zero_data, p_one_data, p_two_data, p_three_data, strain_gauge_zero_data, strain_gauge_one_data)


    
    except Exception as e:
        print("An error occurred:", str(e))
        voltage_task.close()
        strain_task.close()




    return redirect(url_for('index'))