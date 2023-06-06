#include <AccelStepper.h>

// Define the pins for the linear actuator
const uint8_t LINEAR_STEP_PIN = 3;
const uint8_t LINEAR_DIR_PIN = 2;
const uint8_t BOTTOM_LIMIT_SWITCH = 7;
const uint8_t TOP_LIMIT_SWITCH = 8;

// Define the pins for the rotary stepper motor
const uint8_t ROTARY_STEP_PIN = 4;
const uint8_t ROTARY_DIR_PIN = 5;

// Define the resolution of the linear actuator
const float LINEAR_RESOLUTION = 1.8;

// Define the stroke of the linear actuator
const float LINEAR_STROKE = 100.0;

// Define the number of steps per revolution for the linear actuator
const int LINEAR_STEPS_PER_REVOLUTION = 200;

// Define the number of steps per mm for the linear actuator
const float LINEAR_STEPS_PER_MM = LINEAR_STEPS_PER_REVOLUTION / LINEAR_STROKE;

// Define the number of steps per revolution for the rotary stepper motor
const int ROTARY_STEPS_PER_REVOLUTION = 200;

// Define the number of steps per degree for the rotary stepper motor
const float ROTARY_STEPS_PER_DEGREE = ROTARY_STEPS_PER_REVOLUTION / 360.0;

// Define the linear actuator
AccelStepper linearActuator(AccelStepper::DRIVER, LINEAR_STEP_PIN, LINEAR_DIR_PIN);

// Define the rotary stepper motor
AccelStepper rotaryStepper(AccelStepper::DRIVER, ROTARY_STEP_PIN, ROTARY_DIR_PIN);

// Define initial position for the linear actuator
int linPosition = 0;

// Define zero position for the rotary stepper motor
int zeroPosition = 0;

// Variable to store the target position for the linear actuator
int targetLinearPosition = 0;

// Variable to store the target position for the rotary actuator
int targetRotaryPosition = 0;

// Variable to store the return command for the rotary actuator
int returnCommand = 0;

void setup() {
  // Configure the limit switch pins as inputs with pull-up resistors
  pinMode(BOTTOM_LIMIT_SWITCH, INPUT_PULLUP);
  pinMode(TOP_LIMIT_SWITCH, INPUT_PULLUP);

  // Set the maximum speed and acceleration for the linear actuator
  linearActuator.setMaxSpeed(500);
  linearActuator.setAcceleration(400);
  linearActuator.setSpeed(499);

  // Set the maximum speed and acceleration for the rotary stepper motor
  rotaryStepper.setMaxSpeed(500);
  rotaryStepper.setAcceleration(50);
  rotaryStepper.setSpeed(30);

  // Initialize the serial communication
  Serial.begin(9600);
  
  // Set initial position for the linear actuator
  linearActuator.moveTo(500);
  while (linearActuator.distanceToGo() != 0) {
    linearActuator.run();
  }

    while (linearActuator.distanceToGo() != 0){
    linearActuator.run();
  }
  int targetPosition0 = 10000;
  linearActuator.moveTo(targetPosition0);
  while (linearActuator.distanceToGo() != 0) {
      if (digitalRead(TOP_LIMIT_SWITCH) == 1) {
        // Stop the linear actuator when the top limit switch is pressed
        linearActuator.stop();
        linearActuator.setCurrentPosition(0);
        linearActuator.moveTo(200);
        while(linearActuator.distanceToGo() != 0){
          linearActuator.run();
        }
      }
      else if (digitalRead(BOTTOM_LIMIT_SWITCH) == 1) {
      // Stop the linear actuator when the top limit switch is pressed
        linearActuator.stop();
        linearActuator.setCurrentPosition(0);  
        linearActuator.moveTo(-1*targetPosition0); 
      }
      linearActuator.run();
    }

      // Set zero position for the rotary stepper motor
  zeroPosition = rotaryStepper.currentPosition();
}

void loop() {

  // Check if the bottom limit switch is pressed
  // read user input for linear position
  Serial.println("Enter linear position (mm):");
  while (Serial.available() == 0) {
    // Wait for user input
  }
  linPosition = Serial.parseInt();

  // move linear actuator to the specified position
  int targetLinearPosition = linPosition * LINEAR_STEPS_PER_MM;

  linearActuator.moveTo(targetLinearPosition);
 
while (linearActuator.distanceToGo() != 0) {
  // Check if the bottom limit switch is pressed
  if (digitalRead(BOTTOM_LIMIT_SWITCH) == 1) {
    // Move linear actuator to position 0
    linearActuator.stop();
    linearActuator.setCurrentPosition(0); // Reset the actuator's position to 0
    linearActuator.moveTo(-1 * targetLinearPosition);
  }
  // Check if the top limit switch is pressed
  else if (digitalRead(TOP_LIMIT_SWITCH) == 1) {
    // Stop the linear actuator when the top limit switch is pressed
    linearActuator.stop();
    linearActuator.setCurrentPosition(0);
    linearActuator.moveTo(200);
    while (linearActuator.distanceToGo() != 0) {
      linearActuator.run();
    }
  }
    linearActuator.run();
}
  // Check if the linear actuator has reached the target position
  if (linearActuator.distanceToGo() == 0) {
    // Print current linear position
    Serial.print("Current linear position: ");
    Serial.print(linearActuator.currentPosition() / LINEAR_STEPS_PER_MM);
    Serial.println(" mm");

    Serial.println("OK");

    // Read user input for the target position of the rotary actuator
    if (targetLinearPosition != 0) {
      Serial.println("Enter target rotary position (degrees):");
      while (Serial.available() == 0) {
        // Wait for user input
      }
      targetRotaryPosition = Serial.parseInt();

      // Limit the target position to a maximum of 360 degrees
      if (targetRotaryPosition < -360) {
        targetRotaryPosition = -360;
      }
      else if (targetRotaryPosition > 360) {
        targetRotaryPosition = 360;
      }

      // Calculate the target step position based on the zero position
      int targetStepPosition = (targetRotaryPosition * ROTARY_STEPS_PER_DEGREE) + zeroPosition;

      // Move the rotary stepper motor to the specified target position
      rotaryStepper.moveTo(targetStepPosition);

      // Print target position
      Serial.print("Target rotary position: ");
      Serial.print(targetRotaryPosition);
      Serial.println(" degrees");

      // Send confirmation to Python
      Serial.println("Position set successfully");
    }
    else if (returnCommand == 0) {
      // Read user input for the return command
      Serial.println("Enter '0' to return to the original position:");
      while (Serial.available() == 0) {
        // Wait for user input
      }
      returnCommand = Serial.parseInt();

      if (returnCommand == 0) {
        // Move the rotary stepper motor back to the original position
        rotaryStepper.moveTo(zeroPosition);

        // Print original position
        Serial.print("Returning to original position: ");
        Serial.print(zeroPosition / ROTARY_STEPS_PER_DEGREE);
        Serial.println(" degrees");

        // Reset the target position and return command
        targetRotaryPosition = 0;
        returnCommand = -1;

        // Send confirmation to Python
        Serial.println("Returned to original position");
      }
    }

    // Run the rotary stepper motor until it reaches the target position or returns to the original position
    rotaryStepper.run();

    // Check if the rotary stepper motor has reached the target position or the original position
    if (rotaryStepper.distanceToGo() == 0) {
      if (targetRotaryPosition != 0) {
        // Print current position after reaching the target position
        Serial.print("Reached target rotary position: ");
        Serial.print(targetRotaryPosition);
        Serial.println(" degrees");

        // Wait for the return command
        targetRotaryPosition = 0;

        // Send confirmation to Python
        Serial.println("Target position reached");
      }
      else if (returnCommand == 0) {
        // Print current position after returning to the original position
        Serial.print("Returned to original position: ");
        Serial.print(zeroPosition / ROTARY_STEPS_PER_DEGREE);
        Serial.println(" degrees");

        // Reset the return command
        returnCommand = -1;
      }
    }
  }
}
