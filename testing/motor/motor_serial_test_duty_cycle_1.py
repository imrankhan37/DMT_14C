import time
import csv
import serial
import pyvesc
from pyvesc import VESC
import pyvesc
from pyvesc.VESC.messages import  Alive, SetDutyCycle, SetRPM, GetValues

serialport = "COM4"  # Replace "COM4" with the actual port of your VESC

with serial.Serial(serialport, 115200, timeout=0.1) as ser:
    try:
        for i in range(4, 10):
            print(i)
            ser.write(pyvesc.encode(Alive())) # Send heartbeat
            ser.write(pyvesc.encode(SetDutyCycle(i/100)))  # Send command to get values
            ser.write(pyvesc.encode_request(GetValues))


            (response, consumed) = pyvesc.decode(ser.read(100))

            # Decode and process the response
            try:
                decoded_response = pyvesc.decode(response)

                # Extract specific fields
                temp_motor = decoded_response.temp_motor
                duty_cycle_now = decoded_response.duty_cycle_now
                rpm = decoded_response.rpm
                avg_motor_current = decoded_response.avg_motor_current
                avg_input_current = decoded_response.avg_input_current

                print("Temp Motor:", temp_motor)
                print("Duty Cycle Now:", duty_cycle_now)
                print("RPM:", rpm)
                print("Avg Motor Current:", avg_motor_current)
                print("Avg Input Current:", avg_input_current)
                
            except:
                print("Error decoding response")

            time.sleep(5)
        

    except KeyboardInterrupt:
        ser.write(pyvesc.encode(SetRPM(0)))  # Set duty cycle to 0






# filename = "motor_ramp_data.csv"  # Specify the desired filename
# header = ["Speed", "Temperature"]  # Specify the desired header for the CSV columns

# with open(filename, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(header)
#     writer.writerows(data_points)

# print("Data saved to", filename)
# Test script 1: Start the motor at a constant speed, then stop it after 5 seconds

# import serial
# import time
# import csv

# class VESC():
#     def __init__(self, port):
#         self.serial_port = serial.Serial(port, baudrate=115200, timeout=0.1)
#         self.speed = 0
#         self.current = 0
#         self.duty_cycle = 0
#         self.state = 'stopped'

#     def set_duty_cycle(self, duty_cycle):
#         command = f"duty {duty_cycle}\n"
#         self.serial_port.write(command.encode())

#     def get_speed(self):
#         self.serial_port.write(b"get_speed\n")
#         response = self.serial_port.readline().decode().strip()
#         return int(response)

# serialport = "COM4"  # Replace "COM4" with the actual port of your VESC

# data_points = []  # Store the data points in a list

# with serial.Serial(serialport, baudrate=115200, timeout=0.1) as ser:
#     vesc = VESC(ser.port)

#     for i in range(5, 20):
#         try:
#             vesc.set_duty_cycle(i / 100)
#             time.sleep(5)  # Allow time for the motor to stabilize

#             max_attempts = 100
#             attempt = 0
#             while attempt < max_attempts:
#                 speed = vesc.get_speed()
#                 print(speed)
#                 if speed != 0:
#                     data_points.append([i / 100, speed])
#                     break
#                 else:
#                     attempt += 1
#                     continue

#             if attempt >= max_attempts:
#                 print("Exceeded maximum attempts to retrieve speed.")
#                 break

#         finally:
#             vesc.set_duty_cycle(0)

# filename = "motor_ramp_data.csv"  # Specify the desired filename
# header = ["Duty Cycle", "Speed"]  # Specify the desired header for the CSV columns

# with open(filename, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(header)
#     writer.writerows(data_points)

# print("Data saved to", filename)





# try:
#     vesc.start_motor(profile='constant_speed', current=10, duty_cycle=0.53)
#     time.sleep(5)  # Keep the motor running for 5 seconds
# finally:
#     vesc.stop_motor()

# Test script 2: Ramp up the motor speed from 0 to 4000, then ramp it down to 2000, and stop

# vesc = VESC("/dev/ttyUSB0")  # Replace "/dev/ttyUSB0" with the actual port of your VESC

# try:
#     vesc.start_motor(speed=0, profile='ramp_up', current=10, duty_cycle=0.5)
#     time.sleep(5)  # Allow time for ramping up
#     vesc.ramp_down(final_speed=2000)
#     time.sleep(5)  # Allow time for ramping down
# finally:
#     vesc.stop_motor()