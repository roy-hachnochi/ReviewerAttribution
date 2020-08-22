import urllib3
from pathlib import Path
from tqdm import tqdm

# ======================================================================================================================
numReviews = 8035
baseURL = 'https://media.nips.cc/nipsbooks/nipspapers/paper_files/nips31/reviews/'
baseFilename = './dataset_nips/nips31-'

# ======================================================================================================================
# Download raw html files:
connection_pool = urllib3.PoolManager()
for i in tqdm(range(numReviews)):
    url = baseURL + str(i) + '.html'
    resp = connection_pool.request('GET', url)
    data = resp.data.decode("utf-8")
    if data != "Page Not Found":
        filename = Path(baseFilename + str(i) + '.txt')
        filename.touch(exist_ok=True)
        file = open(filename, 'wb')
        file.write(resp.data)
        file.close()
    resp.release_conn()
print("Done.")

