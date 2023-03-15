
PATH_VOXCELEB=

echo "/disk/scratch1/ramons/data/t2s-xling/data/speech/slue-voxceleb/fine-tune_raw" > /disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/train.tsv
cat /disk/scratch1/ramons/data/t2s-xling/data/speech/slue-voxceleb/slue-voxceleb_fine-tune.tsv | awk '{print $1".flac\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' | tail  -n +2 >> /disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/train.tsv


echo "/disk/scratch1/ramons/data/t2s-xling/data/speech/slue-voxceleb/dev_raw" > /disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/valid.tsv
cat /disk/scratch1/ramons/data/t2s-xling/data/speech/slue-voxceleb/slue-voxceleb_dev.tsv | awk '{print $1".flac\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' |  tail  -n +2 >> /disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/valid.tsv


cat /disk/scratch1/ramons/data/t2s-xling/data/speech/slue-voxceleb/slue-voxceleb_dev.tsv /disk/scratch1/ramons/data/t2s-xling/data/speech/slue-voxceleb/slue-voxceleb_fine-tune.tsv | awk '{print $(NF-2)}' | sort | uniq | awk '{print $1" "1}' > /disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/dict.txt

