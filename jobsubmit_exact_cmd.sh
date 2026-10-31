#!/bin/bash
# Submit one exact (QuTiP) or cumulant job to SLURM.
#
# QED_DTC_ROOT  repository root (default: the directory that contains this script)
# CONDA_ENV     environment passed to `conda activate`
#               (default: the NERSC environment used for the paper runs)

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

ROOT="${QED_DTC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CONDA_ENV="${CONDA_ENV:-/global/common/software/py38}"

mkdir -p "$ROOT/msgs"
cd "$ROOT/msgs"

cat > run.sl << EOF
#!/bin/bash
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

conda activate ${CONDA_ENV}

export OMP_NUM_THREADS=1

# mkdir -p \$SCRATCH/qed-dtc/
# cd \$SCRATCH/qed-dtc/

rm -f ~/.cache/matplotlib/fontlist-v330.json

if [ "${method}" = "exact" ]; then
	srun python3 ${ROOT}/src/exact.py ${ep} ${k} ${type_of_coupling} ${lambdaa} ${per} ${N} ${ainit} ${sxinit} ${syinit} ${szinit} ${tmax} ${nphoton} ${evolving} ${dt} ${modulated} ${ntraj} ${i_}
else
	srun python3 ${ROOT}/src/cumulants.py ${ep} ${k} ${type_of_coupling} ${lambdaa} ${per} ${N} ${ainit} ${sxinit} ${syinit} ${szinit} ${method} ${integrator} ${tmax} ${modulated}
fi
EOF

#send interrupt signal 90 seconds before job is terminated
#sbatch --signal=INT@90 run.sl

sbatch run.sl

rm run.sl
