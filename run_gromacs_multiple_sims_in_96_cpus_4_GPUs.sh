#!/bin/bash

GMX_IMG="gromacs_4gpus:2024-3"
TPR="tpr_water_eq.tpr"    # adjust path as needed
JOBS=20
GPUS=4
VCPUS=96
THREADS_PER_JOB=$((VCPUS / JOBS))    # 4 cpu threads/job
#THREADS_PER_JOB=2   # 4 cpu threads/job


for i in $(seq 0 $((JOBS-1))); do
  GPU_IDX=$((i % GPUS))
  echo $GPU_IDX
  OUTDIR="/home/spal/md_$i"
  mkdir -p $OUTDIR
  cp /home/spal/gromacs_md_files/tpr_water_eq.tpr $OUTDIR/

  docker run --rm --user $(id -u):$(id -g) \
  --gpus all -dit \
  -v $OUTDIR:$OUTDIR \
  -w $OUTDIR \
  -e OMP_NUM_THREADS=$THREADS_PER_JOB \
  -e CUDA_VISIBLE_DEVICES=$GPU_IDX \
  $GMX_IMG \
  bash -c "gmx mdrun -s $TPR -deffnm md -ntomp $THREADS_PER_JOB -gpu_id 0 > output.log 2>&1"
done

wait

