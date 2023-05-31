import time
import csv
import serial
import pyvesc
from pyvesc import VESC
from pyvesc.VESC.messages import GetValues, SetDutyCycle, Alive

serialport = "COM4"  # Replace "COM4" with the actual port of your VESC

with serial.Serial(serialport, 115200, timeout=0.05) as ser:
    try:
        for i in range(3,10):
            ser.write(pyvesc.encode(SetDutyCycle(duty=i/100)))

    except KeyboardInterrupt:
        ser.write(pyvesc.encode(SetDutyCycle(duty=0)))



#data_points = []  # Store the data points in a list

# with serial.Serial(serialport, 115200, timeout=0.05) as ser:
#     try:
#         for i in range(5,20):
#             ser.write(pyvesc.encode(SetDutyCycle(duty=i/100)))  # Set duty cycle using raw VESC command
#             ser.write(pyvesc.encode(GetValues()))  # Send command to get values

#             if ser.in_waiting > 61:
#                 (response, consumed) = pyvesc.decode(ser.read(ser.read(61)))

#                 try:
#                     print(response.rpm, response.temp_motor)

#                 except:
#                     print("Error decoding response")
                
#                 time.sleep(0.1)


#     except KeyboardInterrupt:
#         ser.write(pyvesc.encode(SetDutyCycle(duty=0)))  # Set duty cycle to 0



# filename = "motor_ramp_data.csv"  # Specify the desired filename
# header = ["Speed", "Temperature"]  # Specify the desired header for the CSV columns

# with open(filename, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(header)
#     writer.writerows(data_points)

# print("Data saved to", filename)
