# Introduction
cuda operator collections
# Usage
node: any file update requires call cmake to generate new makefile.
```bash
cd rain-ops
# cmake 
cmake -D PROG=matmul
# make, output dir: ./build/bin/${PROG}.out
make
# execute
./${PROG}.out
```
