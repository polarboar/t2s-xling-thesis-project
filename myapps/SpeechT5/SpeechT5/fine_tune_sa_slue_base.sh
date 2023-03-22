

export CUDA_VISIBLE_DEVICES=0

. ./activate.sh


MODEL=speecht5_base
DATA_ROOT=/disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/
SAVE_DIR=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/${MODEL}/slue_sa/
TRAIN_SET=train
VALID_SET=valid
USER_DIR=speecht5
PT_CHECKPOINT_PATH=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/${MODEL}.pt

mkdir -p ${SAVE_DIR}

#  --fp16 \
  #--distributed-world-size 8 \
  #--distributed-port 0 \


fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --hubert-label-dir ${LABEL_DIR} \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  \
  --task speecht5 \
  --t5-task s2t \
  --sample-rate 16000 \
  --num-workers 6 \
  --max-tokens 480256 \
  --update-freq 4 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --max-tokens-valid 3200000 \
  \
  --criterion speecht5 \
  --label-smoothing 0.1 \
  --report-accuracy \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --weight-decay 0.0 \
  --clip-norm 10.0 \
  --lr 0.0002 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 25000 \
  --feature-grad-mult 1.0 \
  \
  --max-update 80000 \
  --max-text-positions 600 \
  --min-speech-sample-size 1056 \
  --max-speech-sample-size 480256 \
  --max-speech-positions 1876 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --keep-last-epochs 10 \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 0 \
  --mask-prob 0.5 \
  --mask-channel-prob 0.5 \
  \
  --finetune-from-model ${PT_CHECKPOINT_PATH}






