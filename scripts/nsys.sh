TIMESTAMP=`date +%Y%m%d_%H%M%S`
NSYS_DIR="/tmp/nsys/"

mkdir -p ${NSYS_DIR}

TIMESTAMP=`date +%Y%m%d_%H%M%S` && nsys profile --stats=true --force-overwrite=true --trace=cuda,nvtx,cublas-verbose,osrt,python-gil --output ${NSYS_DIR}/${TIMESTAMP}_nsys --gpu-metrics-devices=0  "$@"