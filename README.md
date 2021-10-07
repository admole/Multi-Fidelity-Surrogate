# Multi-Fidelity Surrogate

Tests of developing a surrogate model based on multi-fidelity modelling approaches for tandem wall mounted cubes.
 
## Aims

Explore the potential of multi-fidelity modelling approaches to extend the range of data points tested experimentally. This can be achieved by developing a surrogate model of the parametric variation (yaw, offset), so that a continuous variation of key parameters can be obtained with only a limited number of ‘true’ data points. In this instance, fine grid LES data is here taken as the reference ‘truth’. Confidence bounds will be incorporated to represent both uncertainty with the numerical method and data-specific experimental error values. This work will take the following steps:

1. Undertake a series of low-fidelity simulations to provide inaccurate baseline model results. Coarse
LES and RANS will be obtained for several parameter points represent variable fidelity predictions.

2. Split reference data set into training and test and define key target parameters in flow to measure.
Extract data from reference simulations to mimic experimental data and assign confidence levels to
both ‘experimental’ data and modelling predictions. Use Polynomial Chaos Expansion and/or Deep
Gaussian Process Regression to bridge fidelities and obtain surrogate model of parametric variation
(primarily yaw). Compare to test data. Test sensitivity to number and resolution of input simulations.

3. Report on findings including summary of techniques in this field and opportunities for next steps.

## Test Case 

Tandem wall mounted cubes at a Reynolds number of 22,000 is used here as a test case as shown below.

![cubes](Diagrams/tandem_cubes.svg)

This case was chosen because:
- The flow is fully turbulent and exhibits a complex semi-deterministic time-signal in the wake.
- The flow is highly three-dimensional in nature and is beyond the reach of low fidelity models.
- The relative location of the two cubes leads to a non-linear coupling between them.
- A description of the various configurations of cubes used is found [here](Data/README.md).

