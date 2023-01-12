import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pythonNIDAQ
import nidaqmx

from tkinter import Label, Button
import tkinter as Tk

# the following script requires manual setup of virtual hardware in NI MAX software 
# to do: create a script which allows you to input hardware code name, and then create the virtual hardware automatically
    # and then create channels for each of the hardware and then create a task for each of the hardware

# to do: figure out if configurations of the hardware is correct
# to do: write a function to convert output signals to physical units (pressure, temperature, force)
# to do: create a function to smoothen out the data points
# to do: create a GUI to press run and display the data


def get_device_model(device_names):
    """
    Returns the model of an NI DAQ device with the specified name.
    """
    for device_name in device_names:
        try:
            system = nidaqmx.system.System.local()
            for device in system.devices:
                if device.name == device_name:
                    print((f'{device_name}: {device.product_type}'))
        except nidaqmx.DaqError as err:
            print(f'Error: {err}')


def readDAQData(type, device_name, no_of_channels, sample_rate, num_samples, voltage_min, voltage_max):

    # Create instance of nidaqmxWrappers class
    nidaq = pythonNIDAQ.nidaqmxWrappers()

    # Create task
    task = nidaq.returnTaskObj()

    # Add channels to task
    if type == 'voltage':
        task.ai_channels.add_ai_voltage_chan(device_name + '/ai0:' + str(no_of_channels - 1), min_val = voltage_min, max_val = voltage_max)
    elif type == 'temperature':
        task.ai_channels.add_ai_thrmcpl_chan(device_name + '/ai0:' + str(no_of_channels - 1), units=nidaqmx.constants.TemperatureUnits.DEG_C,
                                            thermocouple_type=nidaqmx.constants.ThermocoupleType.K)
    elif type == 'strain':
        task.ai_channels.add_ai_force_bridge_two_point_lin_chan(device_name + '/ai0:' + str(no_of_channels - 1), min_val = 0, max_val = 0.005, physical_units=nidaqmx.constants.BridgePhysicalUnits.KILOGRAM_FORCE, #not sure if these configs are correct
                                                             electrical_units=nidaqmx.constants.BridgeElectricalUnits.MILLIVOLTS_PER_VOLT, bridge_config=nidaqmx.constants.BridgeConfiguration.FULL_BRIDGE)

    # Set sample rate
    task.timing.cfg_samp_clk_timing(sample_rate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

    task.start()
    data = task.read(num_samples)
    task.stop()
    task.close()

    return data


def update_subplots(df, window_size=100):
    # Create the figure and the subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.tight_layout()
    ax1.set_title('Voltage')
    ax2.set_title('Temperature')
    ax3.set_title('Strain')
    plt.xlabel('Time (s)')
    x=df.index[-window_size:]
    # Initialize the lines for each subplot
    voltage_lines = {}
    temperature_lines = {}
    strain_lines = {}

    # Create an empty list to hold the legend handles and labels
    handles, labels = [], []

    def update(num):
        nonlocal x
        num=num+window_size
        x=df.index[num-window_size:num]
        for col in df.columns:
            if col.startswith('Voltage'):
                if col not in voltage_lines:
                    voltage_lines[col], = ax1.plot(x, df[col][num-window_size:num], label=col)
                    handles.append(voltage_lines[col])
                    label  = 'ai' + col[-1] #to get the channel number
                    labels.append(label)
                else:
                    voltage_lines[col].set_data(x, df[col][num-window_size:num])
            elif col.startswith('Temperature'):
                if col not in temperature_lines:
                    temperature_lines[col], = ax2.plot(x, df[col][num-window_size:num], label=col)
                    handles.append(temperature_lines[col])
                    label  = 'ai' + col[-1] #to get the channel number
                    labels.append(label)
                else:
                    temperature_lines[col].set_data(x, df[col][num-window_size:num])
            elif col.startswith('Strain'):
                if col not in strain_lines:
                    strain_lines[col], = ax3.plot(x, df[col][num-window_size:num], label=col)
                    handles.append(strain_lines[col])
                    label  = 'ai' + col[-1] #to get the channel number
                    labels.append(label)
                else:
                    strain_lines[col].set_data(x, df[col][num-window_size:num])
        # update the x and y axis limits
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        ax3.relim()
        ax3.autoscale_view()

        # print(handles)
        # print(labels)

        # Add the legend to the figure
        ax1.legend(handles[:len(voltage_lines)], labels[:len(voltage_lines)])
        ax2.legend(handles[len(voltage_lines):len(voltage_lines)+len(temperature_lines)], labels[len(voltage_lines):len(voltage_lines)+len(temperature_lines)])
        ax3.legend(handles[len(voltage_lines)+len(temperature_lines):], labels[len(voltage_lines)+len(temperature_lines):])

    ani = FuncAnimation(fig, update, frames=range(1, len(df)-window_size+1), repeat=True)
    plt.show()


def main(no_of_voltage_channels, voltage_sampling_rate, no_of_voltage_samples, min_voltage, max_voltage,
        no_of_temperature_channels, temperature_sampling_rate, no_of_temperature_samples,
         no_of_strain_channels, strain_sampling_rate, no_of_strain_samples):

    get_device_model(['Voltage_Measurement', 'Strain_Measurement', 'Temperature_Measurement'])
    
    # Create empty pandas dataframe to store data
    df = pd.DataFrame(columns=['Voltage Measurement 0', 'Voltage Measurement 1',
                                'Voltage Measurement 2', 'Voltage Measurement 3',   
                                'Temperature Measurement 0', 'Temperature Measurement 1',
                                'Strain Measurement 0', 'Strain Measurement 1'])

    # Continuously read data and load into dataframe
    while True:

        # Read data from tasks
        voltage_data = readDAQData('voltage', 'Voltage_Measurement', no_of_voltage_channels, voltage_sampling_rate, no_of_voltage_samples, min_voltage, max_voltage)

        for i in range(0,len(voltage_data)):
            df['Voltage Measurement ' + str(i)] = pd.DataFrame(voltage_data[i])

        temperature_data = readDAQData('temperature', 'Temperature_Measurement', no_of_temperature_channels, temperature_sampling_rate, no_of_temperature_samples, 0, 0)

        for i in range(0, len(temperature_data)):
            df['Temperature Measurement ' + str(i)] = pd.DataFrame(temperature_data[i])

        strain_data = readDAQData('strain', 'Strain_Measurement', no_of_strain_channels, strain_sampling_rate, no_of_strain_samples, 0, 0)

        for i in range(0, len(strain_data)):
            df['Strain Measurement ' + str(i)] = pd.DataFrame(strain_data[i]) 

        update_subplots(df, window_size=100)