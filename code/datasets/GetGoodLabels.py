import csv
import os

# ======================================================================================================================
directory = "./dataset_f1000"
infilename = directory + "/labels5.csv"
outfilename = directory + "/labels.csv"
minShows = 7  # minimum number of appearances to consider label as "good"
labelsInd = 1  # which column of csv is the actual label
# ======================================================================================================================

with open(infilename, mode='r', newline='') as labels_file:
    labels_reader = csv.reader(labels_file, delimiter=',')
    labels_all = [row for row in labels_reader]  # list of lists of all labels

# get actual labels:
filenames = [labels[0] for labels in labels_all]
labels = [labels[labelsInd] for labels in labels_all]

# for all files - check that they aren't empty
valid_inds = []
for i, filename in enumerate(filenames):
    filename = directory + "/" + filename + ".txt"
    if os.stat(filename).st_size != 0:
        valid_inds.append(i)
labels_all = [labels_all[i] for i in valid_inds]
filenames = [filenames[i] for i in valid_inds]
labels = [labels[i] for i in valid_inds]

# create labels histogram, and get good labels:
labelsHist = {}
for label in labels:
    if label in labelsHist:
        labelsHist[label] += 1
    else:
        labelsHist[label] = 1
goodLabels = [(labelsHist[label] >= minShows) for label in labels]

# save new csv:
labels_new = [labels_all[i] for i, isGood in enumerate(goodLabels) if goodLabels[i]]
with open(outfilename, mode='w', newline='') as labels_file:
    labels_writer = csv.writer(labels_file, delimiter=',')
    labels_writer.writerows(labels_new)

print("Number of good labels: {}".format(sum([(count >= minShows) for count in labelsHist.values()])))
print("Number of good data samples: {}".format(len(labels_new)))
