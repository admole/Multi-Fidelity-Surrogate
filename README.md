# Multi-Fidelity Surrogate Modelling of Wall Mounted Cubes

Tests of multi-fidelity surrogate modelling approaches for aerodynamic data of tandem wall mounted cubes.
Details can be found in [this paper](https://doi.org/10.21203/rs.3.rs-2118035/v1).

 A description of the various configurations of cubes used is found [here](Data/README.md).
The data used is stored [here](https://doi.org/10.5281/zenodo.7319243).

## Requirements
The scripts here have been tested with the following main dependencies:
- Python (3.6)
- numpy (1.19.5)
- pandas (1.0.3)
- scikit-learn (0.24.2)
- scikit-optimise (0.9.0)
- matplotlib (3.2.1)

## Retriving data
To retrieve and extract the data use:
```
wget https://zenodo.org/record/7319244/files/Multi_Fidelity_Tandem_Cube_Data.tar.gz
tar -zxf Multi_Fidelity_Tandem_Cube_Data.tar.gz 
```
## Running
To generate selected figures in the paper:
```
cd Plots
./fields.py
./yaw.py
./profiles.py
./slice.py
```
Different command line options are available for each of the scripts to select diffrent outputs or generate animations.
For more information on these, run:
```
./yaw.py -h
./profiles.py -h
./slice.py -h
```

