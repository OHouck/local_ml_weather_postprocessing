#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=mlp_finetune
#SBATCH --account=pi-jfranke
#SBATCH --output=output-%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
 
module load python/3.11.9
source .venv/bin/activate

HOST_IP=`/sbin/ip route get 8.8.8.8 | awk '{print $7;exit}'`
PORT_NUM=$(shuf -i15001-30000 -n1)

echo "=== SSH TUNNEL COMMAND ==="
echo "Run this on your LOCAL machine:"
echo "ssh -L ${PORT_NUM}:${HOST_IP}:${PORT_NUM} ${USER}@<LOGIN_NODE>"
echo ""
echo "Then access: http://localhost:${PORT_NUM}"
echo "=========================="
 
marimo edit --headless --host $HOST_IP --port=$PORT_NUM
 
