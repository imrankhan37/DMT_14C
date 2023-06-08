import threading
import time
import serial
import pyvesc
from pyvesc.VESC.messages import Alive, SetDutyCycle, GetValues

# Create a flag to signal when to stop the motor and reading
experiment_running = True

# Shared variables to store motor values
motor_duty_cycle_recent = None
motor_temp_recent = None

def start_motor(ser):
    try:
        for i in range(5, 15):
            for j in range(5 * 100):
                ser.write(pyvesc.encode(Alive()))  # Send heartbeat
                ser.write(pyvesc.encode(SetDutyCycle(i / 100)))  # Send command to get values
                ser.write(pyvesc.encode_request(GetValues))
                time.sleep(0.01)
    except Exception as e:
        print(f"Error: {str(e)}")

def read_motor_values(ser):
    global motor_duty_cycle_recent
    global motor_temp_recent

    while experiment_running:
        # Check if there is enough data back for a measurement
        if ser.in_waiting > 71:
            (response, consumed) = pyvesc.decode(ser.read(ser.in_waiting))
            # Decode and process the response
            try:
                if response:
                    motor_duty_cycle_recent = response.duty_cycle_now
                    motor_temp_recent = response.temp_fet
                    print(f"Motor duty cycle: {motor_duty_cycle_recent}")
                    print(f"Motor temperature: {motor_temp_recent}")
            except Exception as e:
                print(f"Error processing response: {str(e)}")

# Establish the serial connection
serialport = "COM4"  # Replace "COM4" with the actual port of your VESC
ser = serial.Serial(serialport, 115200, timeout=0.1)

# Create and start the threads
motor_thread = threading.Thread(target=start_motor, args=(ser,))
value_thread = threading.Thread(target=read_motor_values, args=(ser,))

motor_thread.start()
value_thread.start()

# Wait for the threads to finish (optional)
motor_thread.join()
value_thread.join()

# Close the serial connection
ser.close()
