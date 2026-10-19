#!/bin/bash
# Run iswe_pinn_monge.py (E1 in the Monge metric) in background,
# log all console output to txt
cd <workspace>/E1_wellbalance_periodicbed/pinn/code_monge
nohup python3 -u iswe_pinn_monge.py > iswe_pinn_monge_errors.txt 2>&1 &
echo "PID=$!"
sleep 3
ps -p $! -o pid,etime,stat,cmd --no-headers || echo "PROCESS_EXITED"
echo "---LOG_START---"
head -30 iswe_pinn_monge_errors.txt 2>/dev/null
