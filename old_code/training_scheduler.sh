#!/bin/bash
#
#BATCH --job-name=rivers_training
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=amuhebwa@umass.edu
#SBATCH -p titanx-long
#SBATCH --time=07-00:00:00
#SBATCH --mem-per-cpu=10G

module load cudnn/7.6-cuda_9.2

stationschosen="3"
lookBackPeriod=270
orderNumber=6 # 4, 5, 6, 7, 8
nooffeatures=00 # refer to top of columns_utils.py for collect number of features
fileidentifier="Lumped" # AtStation, Lumped or Distributed


#srun -n1 python3 combinations_LSTM_distributed.py --ordernumber=$orderNumber --stationschosen=$stationschosen --nooffeatures=$nooffeatures --fileidentifier=$fileidentifier --lookBackPeriod=$lookBackPeriod
# srun -n1 python3 combinations_LSTM_lumped.py --ordernumber=$orderNumber --stationschosen=$stationschosen --nooffeatures=$nooffeatures --fileidentifier=$fileidentifier --lookBackPeriod=$lookBackPeriod
#srun -n1 python3 combinations_LSTM_atStation.py --ordernumber=$orderNumber --stationschosen=$stationschosen --nooffeatures=$nooffeatures --fileidentifier=$fileidentifier --lookBackPeriod=$lookBackPeriod
