import os
import glob
from tqdm import tqdm

# ======================================================================================================================
folderName = "./dataset_bmj/*.pdf"

# ======================================================================================================================
for filename in tqdm(glob.glob(folderName)):
    outfilename = filename.replace("pdf", "txt")
    os.system("pdftotext {} {}".format(filename, outfilename))
    os.remove(filename)

