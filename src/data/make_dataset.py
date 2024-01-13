# print("Scripts to download or generate data")
import kaggle


def download_data():
    kaggle.api.dataset_download_files(dataset='paultimothymooney/chest-xray-pneumonia', path='./data/external',
                                      force=True,
                                      quiet=False,
                                      unzip=True)
