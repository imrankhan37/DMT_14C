import serial
import time

class ArduinoControl:
    def __init__(self, port):
        self.ser = serial.Serial(port=port, baudrate=9600, timeout=1)

        if not self.ser.is_open:
            self.ser.open()

        time.sleep(2)  # wait for Arduino to initialize

    def move_to(self, rotary_position):
        rotary_position_str = '{:04d}\n'.format(rotary_position)
        self.ser.write(rotary_position_str.encode())

        while True:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().strip().decode()

                if response == 'OK':
                    print('Position set successfully')
                    break
                elif 'ERROR' in response:
                    raise ValueError('Error setting position')
                else:
                    continue  # ignore any other responses

    def close(self):
        self.ser.close()

# Create an instance of the ArduinoControl class
arduino = ArduinoControl('COM3') 

try:
    # Prompt the user for input
    rotary_pos = int(input('Enter rotary position (degrees): '))

    # Move the rotary actuator to the specified position
    arduino.move_to(rotary_pos)

    go_back = int(input('Enter 0 to go back to the original position: '))
    if go_back == 0:
        arduino.move_to(0.00001)

except ValueError as e:
    print('Error:', str(e))

finally:
    # Close the serial connection
    arduino.close()
