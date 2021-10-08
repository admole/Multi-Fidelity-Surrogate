#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 4        # Number of cores (can be 2--32)

module load apps/binapps/paraview/5.6.0

mpiexec -n $NSLOTS pvserver --force-offscreen-rendering

