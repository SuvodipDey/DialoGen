import os
import nltk
import argparse
from nltk.tokenize import word_tokenize
import re

#Post-processes generated dialogue to match the format of reference sentences

parser = argparse.ArgumentParser()
parser.add_argument('-in','--in', help='Name of input file', required=True)
args = vars(parser.parse_args())
input_file = args['in']

output_file = input_file.replace(".txt", "_v2.txt")
print(f"Output file: {output_file}")
f = open(output_file, "w")

with open(input_file, "r") as g:
    for line in g:
        utt_pred1 = line.strip()
        #Post-process output to match references
        pred_tokens = word_tokenize(utt_pred1.lower())
        utt_pred = ' '.join(pred_tokens)
        utt_pred = utt_pred.replace("’","'")
        utt_pred = re.sub("( ')(\s+)*([a-z]+)", r'\1\3', utt_pred)
        utt_pred = re.sub("([a-z]+)(n| n| n |n )(')(t| t)", r"\1 n't", utt_pred)
        utt_pred = re.sub("(\s+)([a-z]+)(\.| \.|\. )([a-z]+)(\s+)", r"\1\2 . \4\5", utt_pred)
        f.write(utt_pred + "\n")
f.close()
print("done")
