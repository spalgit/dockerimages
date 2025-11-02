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


# Create SLURM config directory if it doesn't exist
sudo ln -s /etc/slurm-llnl/slurm.conf /etc/slurm/slurm.conf

# Generate slurm.conf (single node, GPU-enabled)
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
