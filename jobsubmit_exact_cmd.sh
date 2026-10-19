#!/bin/bash

ep=${1}
k=${2}
type_of_coupling=${3}
lambdaa=${4}
per=${5}
N=${6}
ainit=${7}
sxinit=${8}
syinit=${9}
szinit=${10}
method=${11}
integrator=${12}
tmax=${13}
modulated=${14}
evolving=${15}
nphoton=${16}
dt=${17}
ntraj=${18}
mem=${19}
i_=${20}

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
#SBATCH -t 10:00:00

#SBATCH -J exact

module load conda

conda activate /global/common/software/py38

export OMP_NUM_THREADS=1

# mkdir -p $SCRATCH/qed-dtc/
# cd $SCRATCH/qed-dtc/

rm -f ~/.cache/matplotlib/fontlist-v330.json

if [ ${method} = "exact" ]; then
	srun python3 $HOME/qed-dtc/src/exact.py ${ep} ${k} ${type_of_coupling} ${lambdaa} ${per} ${N} ${ainit} ${sxinit} ${syinit} ${szinit} ${tmax} ${nphoton} ${evolving} ${dt} ${modulated} ${ntraj} ${i_}
else
	srun python3 $HOME/qed-dtc/src/cumulants.py ${ep} ${k} ${type_of_coupling} ${lambdaa} ${per} ${N} ${ainit} ${sxinit} ${syinit} ${szinit} ${method} ${integrator} ${tmax} ${modulated}
fi
" > run.sl
#send interrupt signal 90 seconds before job is terminated
#sbatch --signal=INT@90 run.sl

sbatch run.sl

rm run.sl
