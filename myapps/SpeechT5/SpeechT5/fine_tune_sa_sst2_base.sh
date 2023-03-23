

export CUDA_VISIBLE_DEVICES=0

. ./activate.sh


MODEL=speecht5_base
DATA_ROOT=/disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_sst2/
SAVE_DIR=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/${MODEL}/slue_sst2/
TRAIN_SET=train
VALID_SET=valid
USER_DIR=speecht5
PT_CHECKPOINT_PATH=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/${MODEL}.pt

#not sure about this
BPE_TOKENIZER=/disk/scratch2/ramons/data/t2s-xling/models/speechT5/spm_char.model

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
  --hubert-label-dir ${LABEL_DIR} \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  \
  --task speecht5 \
  --t5-task t2c \
  --sample-rate 16000 \
  --num-workers 4 \
  --max-tokens 3200000 \
  --update-freq 1 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --max-tokens-valid 3200000 \
  \
  --criterion speecht5 \
  --use-guided-attn-loss \
  --report-accuracy \
  --sentence-avg \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --dropout 0.15 \
  --activation-dropout 0.15 \
  --attention-dropout 0.15 \
  --encoder-layerdrop 0.0 \
  --decoder-layerdrop 0.0 \
  --weight-decay 0.0 \
  --clip-norm 25.0 \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 \
  --feature-grad-mult 1.0 \
  \
  --max-update 120000 \
  --max-text-positions 600 \
  --min-speech-sample-size 1056 \
  --max-speech-sample-size 480256 \
  --max-speech-positions 1876 \
  --required-batch-size-multiple 1 \
  --skip-invalid-size-inputs-valid-test \
  --keep-last-epochs 10 \
  --validate-after-updates 20000 \
  --validate-interval 50 \
  --log-interval 10 \
  \
  --arch t5_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --freeze-encoder-updates 20000 \
  \
  --finetune-from-model ${PT_CHECKPOINT_PATH}
