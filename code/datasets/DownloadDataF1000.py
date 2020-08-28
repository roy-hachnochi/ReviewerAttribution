import urllib3
from pathlib import Path
from tqdm import tqdm

# ======================================================================================================================
numPages = 9
numReviews = 2500
baseURL = "https://f1000research.com/articles/"
baseFilename = "./dataset_f1000/f1000-"

# ======================================================================================================================
# Download raw html files:
connection_pool = urllib3.PoolManager()
for p in range(7, numPages + 1):
    for i in tqdm(range(numReviews)):
        url = baseURL + str(p) + "-" + str(i) + '.html'
        resp = connection_pool.request('GET', url)
        data = resp.data.decode("utf-8")
        if data != "Page Not Found" and "The page you requested was not found" not in data:
            filename = Path(baseFilename + str(p) + "-" + str(i) + '.txt')
            filename.touch(exist_ok=True)
            file = open(filename, 'wb')
            file.write(resp.data)
            file.close()
        resp.release_conn()
print("Done.")

