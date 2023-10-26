#!/bin/bash -l
#SBATCH --job-name=encodec_training
#SBATCH --clusters=tinygpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=00:5:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV # These 2 commands give var from env

source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python
# module load cuda
# module load cudnn

# Conda
source activate vector-quantize
echo "Job_bash: Activated conda env"

# create a temporary job dir on $TMPDIR
echo "Job_bash: TMPDIR is ${TMPDIR}"
echo "Job_bash: JobID is ${SLURM_JOBID}"
mkdir $TMPDIR/$SLURM_JOBID
cd $TMPDIR/$SLURM_JOBID

# Assign names of data file to variables
TRAIN_FILE="train-clean-100.tar.gz"
TEST_FILE="test-clean.tar.gz"

# COPY THE DATA ON THE NODE
cp $WORK/$TEST_FILE .
cp $WORK/$TRAIN_FILE .
echo "Job_bash: copied the data to node"

mkdir datasets
cd datasets
mkdir data
cd data
echo "Job_bash: created dir for data"

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

# copy input file from location where job was submitted, and run
# cp -r ${SLURM_SUBMIT_DIR}/. .
# mkdir -p output/

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda);"
echo "Job_bash: Begin training"

ENCODEC_SET=encodec320x_ratios8542
DATASET=libri_train100h_test
python train_multi_gpu.py \
                    --config-name=config_HPC \
                    datasets.fixed_length=500 \
                    hydra.run.dir=${WORK}/hydra_outputs/${ENCODEC_SET}_${DATASET}

echo "Job_bash: Finished"

