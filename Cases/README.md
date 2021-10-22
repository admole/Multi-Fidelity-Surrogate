# Cases Setup

This directory contains a base case for the Tandem cubes and a script for creating additional cases.
The definitions of the cases to be created are stored in *.json files an example of which is given by the cases.json file.

## Cases.json
This file contains a list of dictionaries defining the parameters to be modified for each case.
If any of the parameters is not set then a default will be used.
An example entry containing all the parameters that can be changed and their default values is included below:

```json
{
  "Name": "test1",
  "RASModel": "kOmegaSST",
  "Geometry": {
    "Height": 4.0,
    "Width": 9.0,
    "Cube": 1.0,
    "Spacing": 4.0,
    "Length": 14.0,
    "Inlet": 2.0,
    "Offset": 0.0
  },
  "FlowAngle": 0.0,
  "FlowVelocity": 1.0,
  "Ncells": {
    "Nxi": 10,
    "Nxc": 40,
    "Nxb": 25,
    "Nxo": 20,
    "Nyl": 85,
    "Nyu": 15,
    "Nzo": 15,
    "Nzc": 40,
    "Nnc": 75
  },
  "Grading": {
    "ywall": 20.0,
    "yfar": 3.0,
    "xin": 0.4,
    "xc": 1,
    "xb": 1,
    "xo": 10,
    "zc": 1,
    "zr": 7.5,
    "zl": 0.13333,
    "nc": 20,
    "ncm": 0.05
  }
}
```

cases.json contains an example setup for a number of cases.
mesh.json contains the setup for cases used for a mesh resolution study.

## Running

To set up the cases defined in cases.json run:

```bash
chmod +x setup.py
./setup.py
```
or
```bash
python3 setup.py
```

This requires python3 and the packages used to be available.

By default, the cases defined in cases.json are used.
To use other *.json file(s) containing the cases to be set up the files can be specified as argument(s) like:

```bash
python3 setup.py cases.json mesh.json
```