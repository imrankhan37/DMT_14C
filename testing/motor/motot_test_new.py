import csv
import serial
import pyvesc
from pyvesc import VESC

serialport = "COM4"  # Replace "COM4" with the actual port of your VESC

vesc = VESC(serialport)
print(vesc)
