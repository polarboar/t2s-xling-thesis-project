

export CUDA_VISIBLE_DEVICES=0

. ./activate.sh


MODEL=speecht5_base
DATA_ROOT=/disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/
SAVE_DIR=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/${MODEL}/slue_sa_fixed/
TRAIN_SET=train
VALID_SET=valid
USER_DIR=speecht5
PT_CHECKPOINT_PATH=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/${MODEL}.pt

mkdir -p ${SAVE_DIR}

#  --fp16 \
  #--distributed-world-size 2 \
  #--distributed-port 0 \
#original batch_size is 8, we set it lower due to seq length
fairseq-hydra-train \
    --config-dir ./configs \
    --config-name learning_rate

