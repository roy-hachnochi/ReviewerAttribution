import urllib.request
from pathlib import Path
from tqdm import tqdm

def DownloadUrl(url, filename):
    try:
        urllib.request.urlretrieve(url, filename)
        return 1
    except:
        return 0

N_MONTH = 12
N_DAYS = 31

# ======================================================================================================================
if __name__ == '__main__':
    baseURL = 'https://www.bmj.com/sites/default/files/attachments/bmj-article/pre-pub-history/'
    baseFilename = './dataset_bmj/bmj-'
    years = range(20, 15, -1)

    # ======================================================================================================================
    # Download raw html files:
    i = 0
    for year in years:
        for month in tqdm(range(1, N_MONTH + 1)):
            for day in range(1, N_DAYS + 1):
                date = str(day) + '.' + str(month) + '.' + str(year)

                url = baseURL + 'first_decision_' + date + '.pdf'
                filename = Path(baseFilename + str(i) + '.pdf')
                i = i + DownloadUrl(url, filename)

                url = baseURL + 'first_decision_' + date + '_0.pdf'
                filename = Path(baseFilename + str(i) + '.pdf')
                i = i + DownloadUrl(url, filename)

                url = baseURL + 'second_decision_' + date + '.pdf'
                filename = Path(baseFilename + str(i) + '.pdf')
                i = i + DownloadUrl(url, filename)

                url = baseURL + 'second_decision_' + date + '_0.pdf'
                filename = Path(baseFilename + str(i) + '.pdf')
                i = i + DownloadUrl(url, filename)

                url = baseURL + 'thirds_decision_' + date + '.pdf'
                filename = Path(baseFilename + str(i) + '.pdf')
                i = i + DownloadUrl(url, filename)

                url = baseURL + 'thirds_decision_' + date + '_0.pdf'
                filename = Path(baseFilename + str(i) + '.pdf')
                i = i + DownloadUrl(url, filename)

    print("Done.")

