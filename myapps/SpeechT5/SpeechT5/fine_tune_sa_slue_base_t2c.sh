

export CUDA_VISIBLE_DEVICES=0

. ./activate.sh


MODEL=speecht5_base
DATA_ROOT=/home/s2450029/repos/t2s-xling/data_formatted/fairseq/text/tsv/sa_slue/
SAVE_DIR=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/${MODEL}_t2c/slue_sa/
TRAIN_SET=train
VALID_SET=valid
USER_DIR=speecht5
BPE_TOKENIZER=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/spm/spm_char.model
PT_CHECKPOINT_PATH=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/${MODEL}.pt

mkdir -p ${SAVE_DIR}

#  --fp16 \
  #--distributed-world-size 2 \
  #--distributed-port 0 \
#original batch_size is 8, we set it lower due to seq length

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --user-dir ${USER_DIR} \
  --ddp-backend legacy_ddp \
  --log-format json \
  --seed 1 \
  \
  --task speecht5 \
  --t5-task t2c  \
  --num-workers 4 \
  --batch-size 1 \
  --update-freq 16 \
  --data-buffer-size 0 \
  --max-tokens 1400000 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  \
  --criterion speecht5 \
  --report-accuracy \
  --best-checkpoint-metric "s2c_accuracy" \
  --maximize-best-checkpoint-metric \
  \
  --optimizer adam \
  --dropout 0.1 \
  --activation-dropout 0.1 \
  --attention-dropout 0.1 \
  --encoder-layerdrop 0.05 \
  --lr-scheduler triangular \
  --max-lr 2e-4 \
  --lr-period-updates 60000 \
  --lr-shrink 0.5 \
  --lr 1e-8 \
  --feature-grad-mult 1.0 \
  --weight-decay 0.1 \
  \
  --max-update 20000 \
  --max-text-positions 600 \
  --max-speech-positions 250000 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --save-interval-updates 10000 \
  --validate-after-updates 1000 \
  --no-epoch-checkpoints \
  --log-interval 10 \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --mask-prob 0.0 \
  --mask-channel-prob 0.0 \
  --sid-no-pooling-bn \
  --sid-no-embed-postnet \
  \
  --finetune-from-model ${PT_CHECKPOINT_PATH}
