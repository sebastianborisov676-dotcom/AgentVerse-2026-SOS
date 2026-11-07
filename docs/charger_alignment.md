
# Charging-Port Alignment System

## Objective

Control the Franka Panda robotic arm to align a charging connector with a simulated electric vehicle charging port.

## Inputs

- Charging-port position (x, y, z)
- Charging-port orientation
- Robot end-effector position
- Robot end-effector orientation
- Distance to charging port

## Outputs

- Target robot joint positions
- Alignment status
- Alignment error
- Success/failure status

## Development stages

1. Load the Franka Panda model.
2. Create a simulated charging-port target.
3. Read the robot's end-effector coordinates.
4. Calculate positional alignment error.
5. Implement a movement controller.
6. Add orientation alignment.
7. Test with different charging-port positions.
8. Add recovery for failed alignment.

## Success criteria

- Robot reaches the target position within a defined tolerance.
- Robot aligns its connector orientation.
- Robot avoids unsafe or excessive movement.
- Controller reports success or failure.
