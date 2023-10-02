import os
import shutil
import requests
import zipfile
import wget

from paths import DATA

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def funsd():
    print("Downloading FUNSD")

    dlz = DATA / 'funsd.zip'
    download_url("https://guillaumejaume.github.io/FUNSD/dataset.zip", dlz)
    with zipfile.ZipFile(dlz, 'r') as zip_ref:
        zip_ref.extractall(DATA)
    os.remove(dlz)
    os.rename(DATA / 'dataset', DATA / 'FUNSD')

    # adjusted annotations
    aa_train = os.path.join(DATA / 'adjusted_annotations.zip')
    wget.download(url="https://docs.google.com/uc?export=download&id=1cQE2dnLGh93u3xMUeGQRXM81tF2l0Zut", out=aa_train)
    with zipfile.ZipFile(aa_train, 'r') as zip_ref:
        zip_ref.extractall(DATA / 'FUNSD/training_data')
    os.remove(aa_train)
    
    aa_test = os.path.join(DATA / 'adjusted_annotations.zip')
    wget.download(url="https://docs.google.com/uc?export=download&id=18LXbRhjnkdsAvWBFhUdr7_44bfaBo0-v", out=aa_test)
    with zipfile.ZipFile(aa_test, 'r') as zip_ref:
        zip_ref.extractall(DATA / 'FUNSD/testing_data')
    os.remove(aa_test)

    # yolo_bbox
    yolo_train = os.path.join(DATA / 'yolo_bbox.zip')
    
    
    wget.download(url="https://docs.google.com/uc?export=download&id=1UzL5tYtBWDXk_nXj4KtoDMyBt7S3j-aS", out=yolo_train)
    with zipfile.ZipFile(yolo_train, 'r') as zip_ref:
        zip_ref.extractall(DATA / 'FUNSD/training_data')
    os.remove(yolo_train)
    
    yolo_test = os.path.join(DATA / 'yolo_bbox.zip')
    wget.download(url="https://docs.google.com/uc?export=download&id=1fWwbhfvINYFQmoPHpwlH8olyTyJPIe_-", out=yolo_test)
    with zipfile.ZipFile(yolo_test, 'r') as zip_ref:
        zip_ref.extractall(DATA / 'FUNSD/testing_data')
    os.remove(yolo_test)

    return

def get_data():
    funsd()