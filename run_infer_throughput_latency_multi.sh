#!/bin/bash

for (( i=1; i<=$1; i++ )); do
    ./infer_throughput_latency 1 res_multi_proc_p${i}.csv &
done

wait

