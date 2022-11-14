# Test Case 

Tandem wall mounted cubes at a Reynolds number of 22,000 is used here as a test case as shown below.

![cubes](../Diagrams/tandem_cubes.svg)

This case was chosen because:
- The flow is fully turbulent and exhibits a complex semi-deterministic time-signal in the wake.
- The flow is highly three-dimensional in nature and is beyond the reach of low fidelity models.
- The relative location of the two cubes leads to a non-linear coupling between them.


## Methods
the low-fidelity data is obtained using RANS CFD calculations and the high-fidelity using LES CFD calculations.


## Test Configurations

A number of variations of the tandem cube case are used. The parameter that is varied is the yaw angle of the inlet velocity.


## Inlet Yaw
LES Angles: $\theta$ = 0, 5, 10, 15, 20, 25, 30
RANS Angles: $\theta$ = 0 to 40 in increments of 2

![inlet yaw](../Diagrams/inlet_cubes.svg)


