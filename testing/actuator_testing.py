import serial
import time

class ArduinoControl:
    def __init__(self, port):
        self.ser = serial.Serial(port=port, baudrate=9600, timeout=1)

        if not self.ser.is_open:
            self.ser.open()

        time.sleep(2)  # wait for Arduino to initialize

    def move_to(self, linear_position, rotary_position):
        # Send linear position command
        linear_position_str = '{}\n'.format(linear_position)
        self.ser.write(linear_position_str.encode())

        while True:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().strip().decode()

                if response == 'OK':
                    print('Linear position set successfully')
                    break
                else:
                    print('Received:', response)

        # Send rotary position command
        rotary_position_str = '{}\n'.format(rotary_position)
        self.ser.write(rotary_position_str.encode())

        while True:
            if self.ser.in_waiting > 0:
                response = self.ser.readline().strip().decode()

                if response.startswith('Position set'):
                    print(response)
                    break
                elif response.startswith('Error'):
                    raise ValueError(response)
                else:
                    print('Received:', response)

    def close(self):
        self.ser.close()

# Create an instance of the ArduinoControl class
arduino = ArduinoControl('COM3')

try:
    while True:
        try:
            # Prompt the user for input
            lin_pos = float(input('Enter linear position (mm): '))

            # Prompt the user for input
            rotary_pos = int(input('Enter rotary position (degrees): '))

            # Move the linear and rotary actuators to the specified positions
            arduino.move_to(lin_pos, rotary_pos)

            go_back = int(input('Enter 0 to go back to the original position: '))
            if go_back == 0:
                arduino.move_to(0.00001, 0)

        except ValueError as e:
            print('Error:', str(e))

finally:
    # Close the serial connection
    arduino.close()
