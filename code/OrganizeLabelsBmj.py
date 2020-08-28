import csv
import difflib

# ======================================================================================================================
filename = "./dataset_bmj/labels.csv"
thresh = 0.44
# ======================================================================================================================

with open(filename, mode='r', newline='') as labels_file:
    labels_reader = csv.reader(labels_file, delimiter=',')
    labels_all = [row for row in labels_reader]  # list of lists of all labels

# get each label field:
#   0 - file name
#   1 - reviewer name
#   2 - reviewer position
#   3 - reviewer institution
fileNames = [labels[0].lower() for labels in labels_all]
labelsName = [labels[1].lower() for labels in labels_all]
labelsJob = [labels[2].lower() for labels in labels_all]
labelsInstitution = [labels[3].lower() for labels in labels_all]

# sort by reviewer name:
zipped_lists = zip(labelsName, labelsJob, labelsInstitution, fileNames)
sorted_lists = sorted(zipped_lists)
tuples = zip(*sorted_lists)
labelsName, labelsJob, labelsInstitution, fileNames = [list(tuple) for tuple in tuples]

# diff to find similar names:
n_unique_names = 1
for i in range(1, len(labelsName)):
    str1 = labelsName[i - 1]
    str2 = labelsName[i]
    diffs = difflib.ndiff(str1, str2)
    num_diffs = sum(x[0] != ' ' for x in diffs)
    if num_diffs / max(len(str1), len(str2)) <= thresh:  # diff ration is lower than th
        if (str1.split()[0] in str2 and str1.split()[-1] in str2) or (
                str2.split()[0] in str1 and str2.split()[-1] in str1):  # demand that first and last names appear
            labelsName[i] = str1
        else:
            n_unique_names = n_unique_names + 1
    else:
        n_unique_names = n_unique_names + 1
print('Number of unique reviewers: {}, total number of reviews: {}'.format(n_unique_names, len(labelsName)))

# sort back by filename:
zipped_lists = zip(fileNames, labelsName, labelsJob, labelsInstitution)
sorted_lists = sorted(zipped_lists)
tuples = zip(*sorted_lists)
fileNames, labelsName, labelsJob, labelsInstitution = [list(tuple) for tuple in tuples]

# save new csv:
labels_all = [[fileNames[i], labelsName[i], labelsJob[i], labelsInstitution[i]] for i in range(len(labelsName))]
with open(filename, mode='w', newline='') as labels_file:
    labels_writer = csv.writer(labels_file, delimiter=',')
    labels_writer.writerows(labels_all)
