import os
import re
from tqdm import tqdm

# ======================================================================================================================
folderName = "./dataset_nips/"
startToken = '<div style=\"white-space: pre-wrap;\">'
endTokens = ['</div>']

ignores = []  # tokens to ignore
replaces = []  # replacements for ignored tokens

# ======================================================================================================================
# Parse reviews from files:
directory = os.fsencode(folderName)
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    filetxt = filename.split(".")[0]
    if filename.endswith(".txt"):
        with open(folderName + filename, 'r', encoding='utf-8') as oldFile:
            data = oldFile.read()
        if startToken in data:  # parse only files which contain an un-parsed review
            dataSplit = data.split(startToken)[1:]  # first object is html header
            for i, s in enumerate(dataSplit):
                for endToken in endTokens:
                    s = s.split(endToken)[0]  # get review text
                for ig, rep in zip(ignores, replaces):  # remove ignores
                    s = re.sub(ig, rep, s)
                newFilename = filetxt + "-" + str(i) + ".txt"
                with open(folderName + newFilename, 'w+', encoding='utf-8') as newfile:
                    newfile.write(s)
            os.remove(folderName + filename)
        continue
    else:
        continue
print("Done.")

