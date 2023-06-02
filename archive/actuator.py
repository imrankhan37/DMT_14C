import serial
import time

class ArduinoControl:
    def __init__(self, port):
        self.ser = serial.Serial(port=port, baudrate=9600, timeout=1)
        time.sleep(2) # wait for Arduino to initialize
        
    def move_to(self, linear_position, rotary_position):
        # Convert positions to strings with leading zeros and newline characters
        linear_position_str = '{:04d}\n'.format(linear_position)
        rotary_position_str = '{:04d}\n'.format(rotary_position)
        
        # Send positions to Arduino over serial port
        self.ser.write(linear_position_str.encode())
        self.ser.write(rotary_position_str.encode())
        
        # Wait for response from Arduino
        response = self.ser.readline().strip().decode()
        
        # Check for errors
        if response == 'OK':
            print('Positions set successfully')
        elif response == 'ERROR':
            raise ValueError('Error setting positions')
        else:
            raise ValueError('Invalid response from Arduino')
            
    def close(self):
        self.ser.close()