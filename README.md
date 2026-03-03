# Introduction
cuda operator collections
# Usage
node: any file update requires call cmake to generate new makefile.
```bash
cd rain-ops
# cmake 
# if any *.cu files is added, please rerun the cmake cmd
cmake -D PROG=matmul
# make, output dir: ./build/bin/${PROG}.out
make
# execute
./${PROG}.out
```
