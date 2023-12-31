

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

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR}/tb \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --user-dir ${USER_DIR} \
  --ddp-backend legacy_ddp \
  --log-format json \
  --seed 1 \
  \
  --task speecht5 \
  --t5-task s2c \
  --sample-rate 16000 \
  --num-workers 4 \
  --batch-size 2 \
  --update-freq 2 \
  --data-buffer-size 0 \
  \
  --criterion speecht5 \
  --report-accuracy \
  --best-checkpoint-metric "s2c_accuracy" \
  --maximize-best-checkpoint-metric \
  \
  --lr-scheduler fixed \
  --lr 1e-8 \
  \
  --max-update 60000 \
  --max-text-positions 600 \
  --max-speech-positions 8000 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --save-interval 1 \
  --validate-interval 1 \
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
