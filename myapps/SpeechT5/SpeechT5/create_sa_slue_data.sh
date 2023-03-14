
cat /disk/scratch1/ramons/data/t2s-xling/speech/slue-voxceleb/slue-voxceleb_fine-tune.tsv | awk '{print $1"\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' | tail  -n +2 > /disk/scratch1/ramons/data/t2s-xling/speecht5/tsv/sa_slue/train.tsv


cat /disk/scratch1/ramons/data/t2s-xling/speech/slue-voxceleb/slue-voxceleb_dev.tsv | awk '{print $1"\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' |  tail  -n +2 > /disk/scratch1/ramons/data/t2s-xling/speecht5/tsv/sa_slue/valid.tsv

