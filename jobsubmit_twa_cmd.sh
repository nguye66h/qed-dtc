#!/bin/bash

w=${1}
w0=${2}
k=${3}
type_of_coupling=${4}
lambdaa=${5}
per=${6}
N=${7}
ainit=${8}
sxinit=${9}
syinit=${10}
szinit=${11}
tmax=${12}
modulated=${13}
dt=${14}
ntraj=${15}
tasknum=${16}
mem=${17}

mkdir -p $HOME/qed-dtc/msgs/
cd $HOME/qed-dtc/msgs/

echo "#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=cpu
##SBATCH -q debug
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${mem}GB
#SBATCH -t 2:00:00

#SBATCH -J twa

module load conda

conda activate /global/common/software/py38

export OMP_NUM_THREADS=1

# mkdir -p $SCRATCH/qed-dtc/
# cd $SCRATCH/qed-dtc/


srun python3 $HOME/qed-dtc/src/twa.py ${w} ${w0} ${k} ${type_of_coupling} ${lambdaa} ${per} ${N} ${ainit} ${sxinit} ${syinit} ${szinit} ${tmax} ${dt} ${ntraj} ${tasknum} ${modulated}
" > run.sl
#send interrupt signal 90 seconds before job is terminated
#sbatch --signal=INT@90 run.sl

sbatch run.sl

rm run.sl
