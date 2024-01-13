

def download_data():
    import kaggle
    print("Downloading...")
    kaggle.api.dataset_download_files(dataset='paultimothymooney/chest-xray-pneumonia', path='./data/external',
                                      force=True,
                                      quiet=False,
                                      unzip=True)
