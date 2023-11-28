INPUT_FILE=$1
OUTPUT_DIR=$2
MODEL_NAME=$3
CHECKPOINT=$4

# echo "INPUT WAV FILE -> $INPUT_FILE"
before_dot=${INPUT_FILE%%.*}
base_name=$(basename ${before_dot})
# echo "OUTPUT DIR -> $OUTPUT_DIR"


# check the model name is empty or encodec_24khz
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=encodec_24khz
else
    MODEL_NAME=$MODEL_NAME
fi

# check the checkpoint is my_encodec, if so, set the checkpoint path
if [ "$MODEL_NAME" = "my_encodec" ]; then
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT=C:/Study/Thesis/Encodec_checkpoints/bs3_cut36000_length0ep12_lr0.0003.pt
    else
        CHECKPOINT=$CHECKPOINT
    fi
fi

# echo "MODEL NAME -> $MODEL_NAME"

cp $INPUT_FILE $OUTPUT_DIR

TARGET_BANDWIDTH=(1.5 3.0 6.0 12.0 24.0)
for bandwidth in ${TARGET_BANDWIDTH[@]}
do 
    OUTPUT_WAV_FILE=$OUTPUT_DIR/$bandwidth/$base_name.wav
    echo "OUTPUT WAV FILE -> $OUTPUT_WAV_FILE"
    if [ "$MODEL_NAME" = "my_encodec" ]; then
        python main.py -r -b $bandwidth -f $INPUT_FILE $OUTPUT_WAV_FILE -m $MODEL_NAME -c $CHECKPOINT
        echo "Compressed into --> $OUTPUT_WAV_FILE"
    else 
        python main.py -r -b $bandwidth -f $INPUT_FILE $OUTPUT_WAV_FILE -m $MODEL_NAME
    fi
done

read -rsn1 key
# encodec [-r] [-b TARGET_BANDWIDTH] [-f] [--hq] [--lm] $INPUT_FILE $OUTPUT_WAV_FILE