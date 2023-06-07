import pandas as pd
import nidaqmx
import numpy as np
import datetime




voltage_task = None
temperature_task = None
strain_task = None
experiment_running = True  # Flag to track if the experiment is running

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




def read_strain_values():
    global sample_df

    global strain_device
    global strain_channels
    global strain_sampling_rate
    global strain_samples


    # Check if the experiment is not running
    if experiment_running == False:
        return # Exit the function if the experiment is not running

    # Define the channels and parameters for each type of data
    strain_device = 'Strain_Device'
    strain_channels = ['ai1', 'ai2']
    strain_sampling_rate = 200
    strain_samples = 5

    # Create empty pandas dataframe to store data
    data_df = pd.DataFrame(columns=['Strain Measurement {}'.format(i) for i in range(len(strain_channels))])

    # Initialize the DAQ tasks
    tasks = initializeDAQTasks(strain_device=strain_device,
                               strain_channels=strain_channels,
                               strain_sampling_rate=strain_sampling_rate,
                               strain_samples=strain_samples)

    strain_task = tasks['strain']

    time_data = '[]'
    json_strain_gauge_zero_data = '[]'
    json_strain_gauge_one_data = '[]'

    while experiment_running:
        strain_data = readDAQData(strain_task, samples_per_channel=strain_samples, channels=strain_channels,
                                type='strain')
        
        print(strain_data)

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


            # Append the sample dataframe to the data dataframe
            print(sample_df)

        # Append the sample dataframe to the data dataframe
        data_df = pd.concat([data_df, sample_df], ignore_index=True)

        strain_gauge_zero_data = sample_df[['Seconds', 'Strain Measurement 0']]
        strain_gauge_one_data = sample_df[['Seconds', 'Strain Measurement 1']]

        strain1_recent = strain_gauge_zero_data['Strain Measurement 0'].iloc[-1]
        strain2_recent = strain_gauge_one_data['Strain Measurement 1'].iloc[-1]

        print(strain1_recent, strain2_recent)



read_strain_values()