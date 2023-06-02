
import nidaqmx
import numpy as np
import time
import datetime
import pandas as pd
from matplotlib import animation
import os
import threading
from queue import Queue
from flask_socketio import SocketIO, emit
import json

# Define global variables for the tasks
global voltage_task
global temperature_task
global strain_task

def configureDAQ(device_name, type, channels, sampling_rate, samples_per_channel, buffer_size=10000000):
    """
    Configure a DAQ task for a specific type of channel (voltage, temperature, or strain)
    and add the specified channels to the task.
    """
    if type == 'voltage':
        global voltage_task
        voltage_task = nidaqmx.Task()
        task = voltage_task
    elif type == 'temperature':
        global temperature_task
        temperature_task = nidaqmx.Task()
        task = temperature_task
    elif type == 'strain':
        global strain_task
        strain_task = nidaqmx.Task()
        task = strain_task

    # Add the channels to the task
    for channel in channels:
        if type == 'voltage':
            task.ai_channels.add_ai_voltage_chan("{}/{}".format(device_name, channel), min_val=0, max_val=5)
        elif type == 'temperature':
            task.ai_channels.add_ai_thrmcpl_chan("{}/{}".format(device_name, channel),
                                                  thermocouple_type=nidaqmx.constants.ThermocoupleType.K,
                                                  cjc_source=nidaqmx.constants.CJCSource.CONSTANT_USER_VALUE,
                                                  cjc_val=25.0)
        elif type == 'strain':
            task.ai_channels.add_ai_force_bridge_two_point_lin_chan("{}/{}".format(device_name, channel),
                                                                     min_val=-1.0, max_val=1.0,
                                                                     units=nidaqmx.constants.ForceUnits.KILOGRAM_FORCE,
                                                                     bridge_config=nidaqmx.constants.BridgeConfiguration.HALF_BRIDGE,
                                                                     voltage_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL,
                                                                     voltage_excit_val=5, nominal_bridge_resistance=350.0,
                                                                     electrical_units=nidaqmx.constants.BridgeElectricalUnits.MILLIVOLTS_PER_VOLT,
                                                                     physical_units=nidaqmx.constants.BridgePhysicalUnits.KILOGRAM_FORCE)
            
    # Configure the timing of the task
    task.timing.cfg_samp_clk_timing(sampling_rate, samps_per_chan=samples_per_channel, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

    task.in_stream.input_buf_size = buffer_size

def readDAQData(task, samples_per_channel, channels, type):
    """
    Read the data from the specified task and return a dictionary mapping the actual
    channel names to the column data. If the channel is of type 'voltage', it converts
    the data to differential pressure values.
    """
    try:
        # Read the data from the task
        data = task.read(number_of_samples_per_channel=samples_per_channel)

        # Create a dictionary that maps the actual channel names to the column data
        channel_data = {}

        if len(channels) == 1:
            channel_data[channels[0]] = data
        else:
            for i, channel in enumerate(channels):
                if type == 'voltage':
                    # Convert voltage data to differential pressure values
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

def plotData(axs, data, sampling_rate, window_size, voltage_channels = [], temperature_channels = [], strain_channels = []):
    """
    Plot the data for each channel (voltage, temperature, and strain) in the given axs.
    """
    
    # Set the x-axis values based on the sampling rate
    x_values = data['Seconds']

    # Plot the data for each channel
    for i, channel in enumerate(voltage_channels):
        column_name = 'Voltage Measurement {}'.format(i)
        axs[0].plot(x_values, data[column_name], label='Voltage Channel {}'.format(i))

    for i, channel in enumerate(temperature_channels):
        column_name = 'Temperature Measurement {}'.format(i)
        axs[1].plot(x_values, data[column_name], label='Temperature Channel {}'.format(i))

    for i, channel in enumerate(strain_channels):
        column_name = 'Strain Measurement {}'.format(i)
        axs[2].plot(x_values, data[column_name], label='Strain Channel {}'.format(i))

    axs[0].set_ylim(data[[f'Voltage Measurement {i}' for i in range(len(voltage_channels))]].min().min() - 10, data[[f'Voltage Measurement {i}' for i in range(len(voltage_channels))]].max().max() + 10)
    axs[1].set_ylim(data[[f'Temperature Measurement {i}' for i in range(len(temperature_channels))]].min().min() - 1, data[[f'Temperature Measurement {i}' for i in range(len(temperature_channels))]].max().max() + 1)
    axs[2].set_ylim(data[[f'Strain Measurement {i}' for i in range(len(strain_channels))]].min().min(), data[[f'Strain Measurement {i}' for i in range(len(strain_channels))]].max().max())

    # Set the axes limits, labels, and sliding window
    for ax in axs:
        ax.set_xlim(data['Seconds'].iloc[0], data['Seconds'].iloc[-1])
        ax.set_xlabel('Time (s)')
    axs[0].set_ylabel('Pressure (mbar)')
    axs[1].set_ylabel('Temperature (C)')
    axs[2].set_ylabel('Strain')

    # Add legends to the plots
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    # Add gridlines for better visualization
    for ax in axs:
        ax.grid(True)

def findNumberOfChannels(device_name):
    # find the number of channels
    system = nidaqmx.system.System.local()
    for device in system.devices:
        if device.name == device_name:
            number_of_channels = len(device.ai_physical_chans)
            return number_of_channels
        

# write a function to find the device name
def findDeviceName():
    # find the device name
    system = nidaqmx.system.System.local()
    for device in system.devices:
        device_name = device.name
    return device_name

def main(voltage_device='Voltage_DAQ', temperature_device='Temp_Device', strain_device='Strain_Device',
         voltage_channels=['1', '2', '3', '4'], temperature_channels=['1'], strain_channels=['1', '2']):

    # Define the channels and parameters for each type of data
    voltage_channels = ['ai{}'.format(i) for i in range(len(voltage_channels))]
    voltage_sampling_rate = 100
    voltage_samples = 100

    temperature_channels = ['ai{}'.format(i) for i in range(len(temperature_channels))]
    temperature_sampling_rate = 100
    temperature_samples = 100

    strain_channels = ['ai{}'.format(i) for i in range(len(strain_channels))]
    strain_sampling_rate = 100
    strain_samples = 100

    # Create empty pandas dataframe to store data
    data_df = pd.DataFrame(columns=['Voltage Measurement {}'.format(i) for i in range(len(voltage_channels))] +
                                 ['Temperature Measurement {}'.format(i) for i in range(len(temperature_channels))] +
                                 ['Strain Measurement {}'.format(i) for i in range(len(strain_channels))])

    # Configure the DAQ for each type of data
    configureDAQ(device_name=voltage_device, type='voltage', channels=voltage_channels,
                 sampling_rate=voltage_sampling_rate, samples_per_channel=voltage_samples)
    configureDAQ(device_name=temperature_device, type='temperature', channels=temperature_channels,
                 sampling_rate=temperature_sampling_rate, samples_per_channel=temperature_samples)
    configureDAQ(device_name=strain_device, type='strain', channels=strain_channels,
                 sampling_rate=strain_sampling_rate, samples_per_channel=strain_samples)

    while True:
        # Read the data from the DAQ
        voltage_data = readDAQData(voltage_task, samples_per_channel=voltage_samples, channels=voltage_channels,
                                   type='voltage')

        if voltage_data is None:
            return

        temperature_data = readDAQData(temperature_task, samples_per_channel=temperature_samples,
                                       channels=temperature_channels, type='temperature')

        if temperature_data is None:
            return

        strain_data = readDAQData(strain_task, samples_per_channel=strain_samples, channels=strain_channels,
                                  type='strain')

        if strain_data is None:
            return

        # Add the data to the DataFrame
        current_time = datetime.datetime.now()
        num_samples = len(voltage_data[voltage_channels[0]])
        seconds_per_sample = 1.0 / voltage_sampling_rate
        seconds = np.arange(num_samples) * seconds_per_sample

        sample = {'Time': [current_time] * num_samples, 'Seconds': seconds}

        for i, channel in enumerate(voltage_channels):
            column_name = 'Voltage Measurement {}'.format(i)
            sample[column_name] = pd.Series(voltage_data[channel])

        for i, channel in enumerate(temperature_channels):
            column_name = 'Temperature Measurement {}'.format(i)
            sample[column_name] = pd.Series(temperature_data[channel])

        for i, channel in enumerate(strain_channels):
            column_name = 'Strain Measurement {}'.format(i)
            sample[column_name] = pd.Series(strain_data[channel])

        # Convert the sample dictionary to a DataFrame
        sample_df = pd.DataFrame(sample)

        # Append the sample dataframe to the data dataframe
        data_df = data_df.append(sample_df, ignore_index=True)

        #Emit the sample data through the socketIO connection
        #socketIO.emit('data', sample_df.to_json())

        p_zero_data = sample_df[['Seconds', 'Voltage Measurement 0']]
        p_zero_data = p_zero_data.rename(columns={'Seconds': 'Seconds', 'Voltage Measurement 0': 'P_0'})
        json_p_zero_data = p_zero_data.to_json(orient='values')
        print(json_p_zero_data)

        p_one_data = sample_df[['Seconds', 'Voltage Measurement 1']]
        p_one_data = p_one_data.rename(columns={'Seconds': 'Seconds', 'Voltage Measurement 1': 'P_1'})
        json_p_one_data = p_one_data.to_json(orient='values')

        p_two_data = sample_df[['Seconds', 'Voltage Measurement 2']]
        p_two_data = p_two_data.rename(columns={'Seconds': 'Seconds', 'Voltage Measurement 2': 'P_2'})
        json_p_two_data = p_two_data.to_json(orient='values')

        p_three_data = sample_df[['Seconds', 'Voltage Measurement 3']]
        p_three_data = p_three_data.rename(columns={'Seconds': 'Seconds', 'Voltage Measurement 3': 'P_3'})
        json_p_three_data = p_three_data.to_json(orient='values')

        strain_gauge_one_data = sample_df[['Seconds', 'Strain Measurement 0']]
        strain_gauge_one_data = strain_gauge_one_data.rename(columns={'Seconds': 'Seconds', 'Strain Measurement 0': 'Strain_0'})
        json_strain_gauge_one_data = strain_gauge_one_data.to_json(orient='values')

        strain_gauge_two_data = sample_df[['Seconds', 'Strain Measurement 1']]
        strain_gauge_two_data = strain_gauge_two_data.rename(columns={'Seconds': 'Seconds', 'Strain Measurement 1': 'Strain_1'})
        json_strain_gauge_two_data = strain_gauge_two_data.to_json(orient='values')

        return json_p_zero_data, json_p_one_data, json_p_two_data, json_p_three_data, json_strain_gauge_one_data, json_strain_gauge_two_data
