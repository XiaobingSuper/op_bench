export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

#export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
#BATCH_SIZE=64
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=20
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

numactl --physcpubind=0-19 --membind=0 python -u model_benc.py 
