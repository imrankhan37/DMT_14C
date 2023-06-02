# TO DO: 
    # Connect the GUI to the VESC class - when pressing start and stop buttons, the VESC class should be called to start and stop the motor

import pyvesc
import time

class VESC():
    def __init__(self, port):
        # Initialize the pyvesc library with the given port
        self.vesc = pyvesc.VESC(port)
        # Initialize motor parameters
        self.speed = 0
        self.current = 0
        self.duty_cycle = 0
        self.state = 'stopped'

    def start_motor(self, speed, profile, current, duty_cycle):
        # Store the motor parameters
        self.speed = speed
        self.current = current
        self.duty_cycle = duty_cycle
        self.state = 'running'
        # Check that duty cycle, current, and speed are within acceptable ranges
        if duty_cycle < 0 or duty_cycle > 1:
            raise ValueError("Duty cycle must be between 0 and 1")
        if current < 0 or current > 50:
            raise ValueError("Current must be between 0 and 50")
        if speed < 0 or speed > 5000:
            raise ValueError("Speed must be between 0 and 5000")
        # Set the motor configuration parameters using the pyvesc library
        self.vesc.set_duty_cycle(duty_cycle)
        self.vesc.set_current(current)
        self.vesc.set_rpm(speed)
        # Start the motor with the given speed profile
        if profile == 'ramp_up':
            self.ramp_up(speed)
        elif profile == 'ramp_down':
            self.ramp_down(0)
        elif profile == 'constant_speed':
            self.constant_speed()

    def ramp_up(self, final_speed):
        # Incrementally increase the motor speed up to the final speed
        for i in range(self.speed, final_speed, 10):
            self.vesc.set_rpm(i)
            time.sleep(0.1)
            self.speed = i

    def ramp_down(self, final_speed):
        # Incrementally decrease the motor speed down to the final speed
        for i in range(self.speed, final_speed, -10):
            self.vesc.set_rpm(i)
            time.sleep(0.1)
            self.speed = i

    def constant_speed(self):
        # Maintain a constant motor speed
        self.vesc.set_rpm(self.speed)
        while self.state == 'running':
            time.sleep(0.1)

    def stop_motor(self):
        # Stop the motor
        self.state = 'stopped'
        self.vesc.set_rpm(0)

    def check_temp(self, temperature):
        # Check the temperature of the motor and stop it if it exceeds a maximum temperature
        max_temp = 50
        if temperature > max_temp:
            self.stop_motor()
            return True
        else:
            return False
