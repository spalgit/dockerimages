#!/bin/bash
set -e

# Install dependencies and Slurm packages
sudo apt update
sudo apt install -y slurm-wlm munge libmunge2 libmunge-dev

# Create slurm user if not exists
if ! id -u slurm >/dev/null 2>&1; then
    sudo useradd -m slurm
fi

# Setup munge key (generate only if it does not exist)
if [ ! -f /etc/munge/munge.key ]; then
    sudo mungekey
else
    echo "munge key already exists, skipping generation."
fi
sudo chown -R munge: /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key

# Enable and start munge service
sudo systemctl enable munge
sudo systemctl start munge

# Prepare slurm directories
sudo mkdir -p /var/spool/slurmctld /var/log/slurm /var/spool/slurmd
sudo chown slurm: /var/spool/slurmctld /var/log/slurm /var/spool/slurmd

# Create slurm.conf with GPU config
SLURM_CONF="/etc/slurm-llnl/slurm.conf"
sudo tee $SLURM_CONF > /dev/null << EOF
ClusterName=single-node
ControlMachine=localhost
SlurmUser=slurm
SlurmctldPort=6817
SlurmdPort=6818
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SwitchType=switch/none
MpiDefault=none
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
ProctrackType=proctrack/pgid
PluginDir=/usr/lib/x86_64-linux-gnu/slurm-wlm
ReturnToService=2
SlurmctldDebug=info
SlurmdDebug=info
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log

GresTypes=gpu
NodeName=localhost CPUs=4 Gres=gpu:$(nvidia-smi --list-gpus | wc -l) State=UNKNOWN
PartitionName=debug Nodes=localhost Default=YES MaxTime=INFINITE State=UP
EOF

sudo chown slurm: $SLURM_CONF
sudo chmod 644 $SLURM_CONF

# Create gres.conf describing GPU devices
GRES_CONF="/etc/slurm-llnl/gres.conf"
sudo tee $GRES_CONF > /dev/null << EOF
Name=gpu Type=tesla File=/dev/nvidia0
EOF

sudo chown slurm: $GRES_CONF
sudo chmod 644 $GRES_CONF

# Enable and start slurm controller and daemon
sudo systemctl enable slurmctld
sudo systemctl enable slurmd
sudo systemctl start slurmctld
sudo systemctl start slurmd

