import os
import csv

# ======================================================================================================================
folderName = "./dataset_bmj/"
ignore = "<*>"
startToken = 'Comments:\n'
endToken = '\nAdditional Questions:'

nameToken = "Please enter your name: "
jobToken = "Job Title: "
institutionToken = "Institution: "

# ======================================================================================================================
# Parse reviews from files, and get labels:
labelsName = []
labelsJob = []
labelsInstitution = []
fileNames = []
directory = os.fsencode(folderName)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filetxt = filename.split(".")[0]
    if filename.endswith(".txt"):
        with open(folderName + filename, 'r', encoding='utf-8') as oldFile:
            data = oldFile.read()
        if startToken in data:  # parse only files which contain an un-parsed review
            dataSplit = data.split(startToken)[1:]  # first object isn't a review
            for i, s in enumerate(dataSplit):
                if nameToken in s and jobToken in s and institutionToken in s:
                    # get labels:
                    labelsName.append(s.split(nameToken)[1].split("\n")[0])
                    labelsJob.append(s.split(jobToken)[1].split("\n")[0])
                    labelsInstitution.append(s.split(institutionToken)[1].split("\n")[0])
                    fileNames.append(filetxt + "-" + str(i))
                    # get review text:
                    sNew = s.split(endToken)[0]
                    sNew = sNew.replace(ignore, "")
                    newFilename = filetxt + "-" + str(i) + ".txt"
                    with open(folderName + newFilename, 'w+', encoding='utf-8') as newfile:
                        newfile.write(sNew)
        os.remove(folderName + filename)
        continue
    else:
        continue

# save labels:
labels_all = [[fileNames[i], labelsName[i], labelsJob[i], labelsInstitution[i]] for i in range(len(labelsName))]
labels_all = [[''.join(i for i in s if ord(i) < 128) for s in l] for l in labels_all]  # ignore non-ascii chars
with open(folderName + "lables.csv", mode='w', newline='') as labels_file:
    labels_writer = csv.writer(labels_file, delimiter=',')
    labels_writer.writerows(labels_all)
print("Done.")

