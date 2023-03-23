import sys
import sentencepiece as spm



input_path=sys.argv[1]
model_path=sys.argv[2]
output_path=sys.argv[3]


sp = spm.SentencePieceProcessor()
sp.load(model_path)

with open(input_path) as inputfile, open(output_path,"w") as outputfile:
    for line in inputfile.readlines():
        tokens = sp.encode_as_pieces(line.strip())
        outputfile.write(" ".join(tokens)+"\n")


