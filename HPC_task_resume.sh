#!/bin/bash -l
#SBATCH --job-name=encodec_training_resume_360
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --output=LOG_%x.%j.out
#SBATCH --error=LOG_%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=24:00:00
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

# create a temporary job dir on $TMPDIR
echo "Job_bash: TMPDIR is ${TMPDIR}"
echo "Job_bash: JobID is ${SLURM_JOBID}"
mkdir $TMPDIR/$SLURM_JOBID
cd $TMPDIR/$SLURM_JOBID

# Assign names of data file to variables
TRAIN_FILE="train-clean-360.tar.gz"
TEST_FILE="test-clean.tar.gz"

# COPY THE DATA from WORK to the Node at $TMPDIR/$SLURM_JOBID
cp $WORK/$TEST_FILE .
cp $WORK/$TRAIN_FILE .
echo "Job_bash: copied the data to node"

# create dir for data
mkdir datasets
cd datasets
mkdir data
cd data

# Now in $TMPDIR/$SLURM_JOBID/datasets/data
tar -xzf $TMPDIR/$SLURM_JOBID/$TEST_FILE
echo "Job_bash: unpacked test"
tar -xzf $TMPDIR/$SLURM_JOBID/$TRAIN_FILE
echo "Job_bash: unpacked train"
# Data lies at $TMPDIR/$SLURM_JOBID/datasets/data/LibriSpeech

# Now in $TMPDIR/$SLURM_JOBID/datasets
cd ../
# Generate descriptive csv files
source $HOME/Encodec/datasets/generate_csv.sh

cd $HOME/Encodec

echo "Job_bash: Begin training"

ENCODEC_SET=encodec320x_ratios8542
DATASET=libri_train360h_test
python train_multi_gpu.py \
                    --config-name=config_HPC_resume \
                    hydra.run.dir=${WORK}/hydra_outputs/${SLURM_JOBID}/${ENCODEC_SET}_${DATASET}
                     # datasets.fixed_length=500 \

echo "Job_bash: Finished"

