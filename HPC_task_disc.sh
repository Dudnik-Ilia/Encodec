#!/bin/bash -l
#SBATCH --job-name=discriminator_training
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx3080
#SBATCH --output=LOG_%x.%j.out
#SBATCH --error=LOG_%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=12:00:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV # These 2 commands give var from env

source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# module purge
# module load python/3.10-anaconda
# module load cuda
# module load cudnn

# Conda
source activate encodec_copy
echo "Job_bash: Activated conda env: ${CONDA_DEFAULT_ENV}"

python -c "import torch; print('Cuda is_available: ',torch.cuda.is_available()); print('Cuda version: ',torch.version.cuda);"

if ! python -c "import torch; torch.cuda.is_available()"; then
    echo "CUDA is not available. Exiting the batch job."
    exit 1
fi

# create a temporary job dir on 2dw$TMPDIR
echo "Job_bash: TMPDIR is ${TMPDIR}"
echo "Job_bash: JobID is ${SLURM_JOBID}"
mkdir $TMPDIR/$SLURM_JOBID

cd $HOME/Encodec

echo "Job_bash: Begin training"

python train_discriminator.py \
                    --config-name=config_HPC_disc \
                    hydra.run.dir=${WORK}/hydra_outputs/${SLURM_JOBID}/
                     # datasets.fixed_length=500 \

echo "Job_bash: Finished"

