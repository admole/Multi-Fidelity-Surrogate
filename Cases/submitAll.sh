#!/bin/bash

for name in *
do
  if [ -d "$name" ] && [ "$name" != "Base" ]
  then
    echo "Running $name"
    cd "$name" ||exit
    sed -i "s/Base/$name/g" runBoth
    qsub runBoth
    cd ../
  fi
done
