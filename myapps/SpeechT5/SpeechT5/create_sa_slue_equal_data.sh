
PATH_VOXCELEB=/home/s2450029/repos/slue-toolkit/data/slue-voxceleb
PATH_DESTINATION=/home/s2450029/repos/t2s-xling/data_formatted/fairseq/speech/tsv/sa_slue

#echo $PATH_VOXCELEB/fine-tune_raw
#echo "$PATH_DESTINATION/train.tsv"



echo "$PATH_VOXCELEB/fine-tune_raw" > $PATH_DESTINATION/train.tsv
cat $PATH_VOXCELEB/slue-voxceleb_fine-tune.tsv | awk -v FS='\t' '{if($5!="<mixed>")print $1".flac\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' | tail  -n +2 > $PATH_DESTINATION/train_temp.tsv


# Calculate the counts of each type (Negative, Neutral, Positive)
count_negative=$(awk -v FS='\t' '$3 == "Negative" { count++ } END { print count }' "$PATH_DESTINATION/train_temp.tsv")
count_neutral=$(awk -v FS='\t' '$3 == "Neutral" { count++ } END { print count }' "$PATH_DESTINATION/train_temp.tsv")
count_positive=$(awk -v FS='\t' '$3 == "Positive" { count++ } END { print count }' "$PATH_DESTINATION/train_temp.tsv")

# Calculate the maximum count among all types
max_count=$(printf "%d\n" "$count_negative" "$count_neutral" "$count_positive" | sort -nr | head -n 1)

# Make sample from all classes equal
awk -v max_count=$max_count -v count_positive=$count_positive -v count_negative=$count_negative -v count_neutral=$count_neutral '{
    type = $NF;
    if (type == "Neutral") {
	print $0
    }
    else if (type == "Positive") {
	    for (i = 0; i < int(max_count/count_positive); i++) {
	    	print $0
	    }
    }
    else if (type == "Negative") {
	    for (i = 0; i < int(max_count/count_negative); i++) {
		print $0
	}
    }
}' $PATH_DESTINATION/train_temp.tsv >> $PATH_DESTINATION/train.tsv
rm $PATH_DESTINATION/train_temp.tsv

echo "$PATH_VOXCELEB/dev_raw" > $PATH_DESTINATION/valid.tsv
cat $PATH_VOXCELEB/slue-voxceleb_dev.tsv | awk -v FS='\t' '{if($5!="<mixed>" && $5!="Disagreement")print $1".flac\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' |  tail  -n +2 >> $PATH_DESTINATION/valid.tsv

#cat $PATH_VOXCELEB/slue-voxceleb_dev.tsv $PATH_VOXCELEB/slue-voxceleb_fine-tune.tsv | awk -v FS='\t' '{if($5!="<mixed>" && $5!="Disagreement" && $5!="sentiment")print $(NF-2)}' | sort | uniq | awk '{print $1" "1}' > $PATH_DESTINATION/dict.txt

