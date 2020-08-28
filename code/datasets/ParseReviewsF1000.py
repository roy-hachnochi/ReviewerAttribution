import os
import re
import csv
from tqdm import tqdm

def get_text(data, startToken, endTokens, ignores, replaces):
    if startToken not in data:
        return []
    sList = []
    dataSplit = data.split(startToken)[1:]
    for s in dataSplit:
        # get text:
        for endToken in endTokens:
            s = s.split(endToken)[0]  # get review text
        for ig, rep in zip(ignores, replaces):  # remove ignores
            s = re.sub(ig, rep, s)
        s = re.sub(" +", " ", s)  # remove multiple spaces
        sList.append(s)
    return sList

# ======================================================================================================================
folderName = "./dataset_f1000/"
revStartToken = '<span class="f1r-icon icon-14_more_small orange vmiddle big"></span> </button> </div> </span> <span class="hidden hidden-report-text"> <div class=hidden-report-text__comment>'
revEndTokens = ['</div> <p style="margin-top: 10px;">', '<div class=md>']

labelStartToken = "<span class=\"info-separation padding-bottom padding-left\"> <div> <span class=bold>"
labelEndTokens = ["</span><span class=f1r-article-desk-inline>"]

ignores = [r'<.*?>', '&quot;', '&nbsp;', r'&\S*;']  # tokens to ignore
replaces = [" ", "\"", " ", ""]  # replacements for ignored tokens

# ======================================================================================================================
# Parse reviews from files, and get labels:
fileNames = []
labels = []
directory = os.fsencode(folderName)
for file in tqdm(os.listdir(directory)):
    filename = os.fsdecode(file)
    filetxt = filename.split(".")[0]
    if filename.endswith(".txt"):
        with open(folderName + filename, 'r', encoding='utf-8') as oldFile:
            data = oldFile.read()
        revList = get_text(data, revStartToken, revEndTokens, ignores, replaces)
        labelsList = get_text(data, labelStartToken, labelEndTokens, ignores, replaces)
        if len(revList) != 0:  # parse only files which contain an un-parsed review
            for i, rev in enumerate(revList):
                newFilename = filetxt + "-" + str(i) + ".txt"
                with open(folderName + newFilename, 'w+', encoding='utf-8') as newfile:
                    newfile.write(rev)
                fileNames.append(filetxt + "-" + str(i))
                labels.append(labelsList[i])
            os.remove(folderName + filename)
        continue
    else:
        continue
print("Done.")

# save labels:
labels_all = [[fileNames[i], labels[i]] for i in range(len(labels))]
labels_all = [[''.join(i for i in s if ord(i) < 128) for s in l] for l in labels_all]  # ignore non-ascii chars
with open(folderName + "labels.csv", mode='w', newline='') as labels_file:
    labels_writer = csv.writer(labels_file, delimiter=',')
    labels_writer.writerows(labels_all)
print("Done.")