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

interval = 100  # Specify the interval between data points in milliseconds
data = []
start_time = time.time()  # Store the start time

fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('RPM')

# Initialize the plot with empty data
line, = ax.plot([], [], 'b-')

def update_plot(x, y):
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()

with serial.Serial(serialport, 115200, timeout=0.1) as ser:
    try:
        for i in range(4000, 8000, 500):

            for j in range(interval):

                ser.write(pyvesc.encode(Alive()))  # Send heartbeat
                ser.write(pyvesc.encode(SetRPM(i)))  # Send command to get values
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
                                'avg_motor_current': response.avg_motor_current,
                                'avg_input_current': response.avg_input_current,
                                'amp_hours': response.amp_hours * 1000
                            })

                            # Extract the required values for plotting
                            x = [point['time'] for point in data]
                            y = [point['rpm'] for point in data]

                            # Update the plot
                            update_plot(x, y)

                    except Exception as e:
                        print(f"Error processing response: {str(e)}")

        # Convert data to JSON and print
        json_data = json.dumps(data)
        print(json_data)

    except Exception as e:
        print(f"Error: {str(e)}")

    except KeyboardInterrupt:
        ser.write(pyvesc.encode(SetRPM(0)))  # Set duty cycle to 0

plt.show()
