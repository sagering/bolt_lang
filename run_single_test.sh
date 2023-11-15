#!/bin/bash

echo
echo Test: $1

python ./compiler.py "./tests/$1.bolt" "./build/$1.cpp"
docker run -v $(pwd -W):/home/ --rm gcc:10.2 g++ ./home/build/$1.cpp -o ./home/build/$1 -lc
docker run -v $(pwd -W):/home/ --rm gcc:10.2 sh -c ./home/build/$1