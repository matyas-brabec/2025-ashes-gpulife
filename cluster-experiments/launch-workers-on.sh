#!/bin/bash

PARTITION=$1
SCRIPT_TEMPLATE=$2
WORKER_COUNT=$3

should_exit=0

if [ -z "$PARTITION" ]; then
    echo "param \$1: Partition not provided"
    should_exit=1
fi

if [ -z "$SCRIPT_TEMPLATE" ]; then
    echo "param \$2: Script template not provided"
    should_exit=1
fi

if [ -z "$WORKER_COUNT" ]; then
    echo "param \$3: Worker count not provided"
    should_exit=1
fi

if [ $should_exit -eq 1 ]; then
    echo "Exiting..."
    exit 1
fi

slurm_script="./internal-scripts/_${PARTITION}.job.slurm.sbatch.sh"

if [ ! -d "../build-stable" ]; then
    echo "stable build not found (folder name: build-stable)"
    exit 1
fi

stable_exe_crated_at=$(ls -l ../build-stable/src/stencils | awk '{print $6 " " $7 " " $8}')
dev_exe_crated_at=$(ls -l ../build/src/stencils | awk '{print $6 " " $7 " " $8}')

stable_exe_crated_at_epoch=$(date -d "$stable_exe_crated_at" +%s)
dev_exe_crated_at_epoch=$(date -d "$dev_exe_crated_at" +%s)

time_difference=$((dev_exe_crated_at_epoch - stable_exe_crated_at_epoch))
# time_difference=$((stable_exe_crated_at_epoch - dev_exe_crated_at_epoch))
abs_time_difference=${time_difference#-}

echo
echo "Time difference between stable and dev build:"
echo -e "\e[33m$(eval echo `date -d @$((abs_time_difference)) -u +'$((%d - 01)) days, %Hh %Mm'`)\e[0m"

if [ $time_difference -lt 0 ]; then
    echo -e "\e[36mDev build is \e[1;32mOLDER\e[0m\e[36m than stable build\e[0m"
else
    echo -e "\e[36mDev build is \e[1;31mNEWER\e[0m\e[36m than stable build\e[0m"
fi 

echo
read -p "Do you wish to proceed with the current stable build? (y/n): " -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Exiting..."
    exit 1
fi

echo "launching $slurm_script on $WORKER_COUNT workers"
echo

for i in $(seq 1 $WORKER_COUNT)
do
    echo launching $slurm_script $SCRIPT_TEMPLATE $i 
    sbatch $slurm_script $SCRIPT_TEMPLATE $i 
    echo
done