import time
import csv
import serial
import pyvesc
import json
import matplotlib.pyplot as plt
from pyvesc import VESC
import pyvesc
from pyvesc.VESC.messages import Alive, SetDutyCycle, SetRPM, GetValues

serialport = "COM4"  # Replace "COM4" with the actual port of your VESC

interval = 50  # Specify the interval between data points in milliseconds
data = []
start_time = time.time()  # Store the start time

fig, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].set_xlabel('Time')
axs[0].set_ylabel('RPM')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Duty Cycle')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Temperature')

# Initialize the plots with empty data
line_rpm, = axs[0].plot([], [], 'b-', label='RPM')
line_duty_cycle, = axs[1].plot([], [], 'r-', label='Duty Cycle')
line_temperature, = axs[2].plot([], [], 'g-', label='Temperature')

def update_plot(x, y, line):
    line.set_data(x, y)
    line.axes.relim()
    line.axes.autoscale_view()
    fig.canvas.draw()

with serial.Serial(serialport, 115200, timeout=0.1) as ser:

    try:
        for i in range(6, 15):
            for j in range(interval):
                ser.write(pyvesc.encode(Alive()))  # Send heartbeat
                ser.write(pyvesc.encode(SetDutyCycle(i / 100)))  # Send command to get values
                ser.write(pyvesc.encode_request(GetValues))
                time.sleep(0.01)

                # Check if there is enough data back for a measurement
                if ser.in_waiting > 71:
                    (response, consumed) = pyvesc.decode(ser.read(ser.in_waiting))
                    # Decode and process the response
                    try:
                        if response:
                            elapsed_time = time.time() - start_time  # Calculate elapsed time
                            data.append({
                                'time': elapsed_time,
                                'duty_cycle_now': response.duty_cycle_now,
                                'rpm': response.rpm,
                                'motor_temp': response.temp_fet,
                                # temp_motor
                                'avg_motor_current': response.avg_motor_current,
                                'avg_input_current': response.avg_input_current,
                                'amp_hours': response.amp_hours * 1000
                            })

                            # Extract the required values for plotting
                            x = [point['time'] for point in data]
                            y_rpm = [point['rpm'] for point in data]
                            y_duty_cycle = [point['duty_cycle_now'] for point in data]
                            y_temperature = [point['motor_temp'] for point in data]

                            # Update the plots
                            update_plot(x, y_rpm, line_rpm)
                            update_plot(x, y_duty_cycle, line_duty_cycle)
                            update_plot(x, y_temperature, line_temperature)

                    except Exception as e:
                        print(f"Error processing response: {str(e)}")

        # Convert data to JSON and print
        json_data = json.dumps(data)
        print(json_data)

    except Exception as e:
        print(f"Error: {str(e)}")

    except KeyboardInterrupt:
        ser.write(pyvesc.encode(SetRPM(0)))

plt.show()

