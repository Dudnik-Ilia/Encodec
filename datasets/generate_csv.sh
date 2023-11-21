#!/bin/bash
# Create a csv description files for train and test
# Input: This bash file is called from HPC task bash file where $TRAIN_FILE and $TEST_FILE are already specified
# Output: Created csv files with names "librispeech_dataset_train" and "librispeech_dataset_test"
TRAIN_WITHOUT_EXT=$(basename "$TRAIN_FILE" .tar.gz)
TEST_WITHOUT_EXT=$(basename "$TEST_FILE" .tar.gz)

input_file="$TMPDIR/$SLURM_JOBID/datasets/data/LibriSpeech/$TRAIN_WITHOUT_EXT"
output_file="$TMPDIR/$SLURM_JOBID/datasets/librispeech_dataset_train.csv"
python $HOME/Encodec/datasets/generate_desc_file.py -i $input_file -o $output_file
echo "Generated TRAIN csv file"


input_file="$TMPDIR/$SLURM_JOBID/datasets/data/LibriSpeech/$TEST_WITHOUT_EXT"
output_file="$TMPDIR/$SLURM_JOBID/datasets/librispeech_dataset_test.csv"
python $HOME/Encodec/datasets/generate_desc_file.py -i $input_file -o $output_file
echo "Generated TEST csv file"