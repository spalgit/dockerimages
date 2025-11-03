#!/bin/bash
set -e

# Install dependencies and Slurm packages
sudo apt update
sudo apt install -y slurm-wlm munge libmunge2 libmunge-dev

# Create slurm user if not exists
if ! id -u slurm >/dev/null 2>&1; then
    sudo useradd -m slurm
fi

# Setup munge key if missing
if [ ! -f /etc/munge/munge.key ]; then
    sudo /usr/sbin/mungekey --force
else
    echo "munge key already exists, skipping generation."
fi
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key

# Enable and start munge service
sudo systemctl enable munge
sudo systemctl start munge

# Prepare slurm directories
#sudo mkdir -p /var/spool/slurmctld /var/log/slurm /var/spool/slurmd /usr/lib/x86_64-linux-gnu/slurm-wlm
#sudo chown slurm: /var/spool/slurmctld /var/log/slurm /var/spool/slurmd

sudo apt update -y
sudo apt install slurmd slurmctld -y
sudo mkdir /etc/slurm-llnl/
sudo chmod 777 /etc/slurm-llnl
sudo mkdir /var/lib/slurm-llnl/
sudo mkdir /var/log/slurm-llnl/
sudo chmod 777 /var/lib/slurm-llnl/
sudo chmod 777 /var/log/slurm-llnl/

CPU_INFO=$(lscpu)
CPUS=$(echo "$CPU_INFO" | awk '/^CPU\(s\):/ {print $2}')
SOCKETS=$(echo "$CPU_INFO" | awk '/Socket\(s\):/ {print $2}')
CORES_PER_SOCKET=$(echo "$CPU_INFO" | awk '/Core\(s\) per socket:/ {print $4}')
THREADS_PER_CORE=$(echo "$CPU_INFO" | awk '/Thread\(s\) per core:/ {print $4}')
# If lscpu output uses slightly different fields, adjust awk patterns accordingly.
REALMEM=$(free -m | awk '/^Mem:/ { print int($2 * 0.95) }')
BOARDS=1  # Standard for x86 cloud VMs

# Create SLURM config directory if it doesn't exist
sudo ln -s /etc/slurm-llnl/slurm.conf /etc/slurm/slurm.conf

# Generate slurm.conf (single node, GPU-enabled)
SLURM_CONF="/etc/slurm-llnl/slurm.conf"
sudo tee $SLURM_CONF > /dev/null << EOF
ClusterName=localcluster
SlurmctldHost=localhost
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/var/run/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/var/lib/slurm-llnl/slurmd
SlurmUser=slurm
StateSaveLocation=/var/lib/slurm-llnl/slurmctld
SwitchType=switch/none
TaskPlugin=task/none
#
# TIMERS
InactiveLimit=0
KillWait=30
MinJobAge=300
SlurmctldTimeout=120
SlurmdTimeout=300
Waittime=0
# SCHEDULING
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
#
#AccountingStoragePort=
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/none
JobAcctGatherFrequency=30
JobAcctGatherType=jobacct_gather/none
SlurmctldDebug=info
SlurmctldLogFile=/var/log/slurm-llnl/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/var/log/slurm-llnl/slurmd.log
#
# COMPUTE NODES
NodeName=localhost CPUs=$CPUS Boards=$BOARDS Sockets=$SOCKETS CoresPerSocket=$CORES_PER_SOCKET ThreadsPerCore=$THREADS_PER_CORE RealMemory=$REALMEM
PartitionName=LocalQ Nodes=ALL Default=YES MaxTime=INFINITE State=UP

GresTypes=gpu
EOF


sudo chown slurm: $SLURM_CONF
sudo chmod 644 $SLURM_CONF

# Generate gres.conf for GPU resources (assumes Tesla, adjust as needed)
GRES_CONF="/etc/slurm-llnl/gres.conf"
gpu_count=$(nvidia-smi --list-gpus | wc -l)
sudo tee $GRES_CONF > /dev/null << EOF
$(for i in $(seq 0 $((gpu_count-1))); do echo "Name=gpu Type=tesla File=/dev/nvidia$i"; done)
EOF

sudo chown slurm: $GRES_CONF
sudo chmod 644 $GRES_CONF

# Enable and start slurm controller and daemon
sudo systemctl enable slurmctld
sudo systemctl enable slurmd
sudo systemctl start slurmctld
sudo systemctl start slurmd

sudo service slurmctld restart && sudo service slurmd restart
