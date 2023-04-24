"""
Downloads the following:
- Stanford parser
- Stanford POS tagger
- Glove vectors
- SICK dataset (semantic relatedness task)
- Stanford Sentiment Treebank (sentiment classification task)

"""

import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import requests
from tqdm import tqdm


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
            self.pbar.set_description("Downloading")

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded - self.pbar.n)
        else:
            self.pbar.close()
            self.pbar = None


def download(url, dirpath, overwrite=False) -> Path:
    filename = url.split('/')[-1]
    filepath = Path(dirpath).joinpath(filename)

    req = requests.head(url, allow_redirects=True)
    file_size = int(req.headers["Content-Length"])
    if filepath.exists() and filepath.stat().st_size == file_size and not overwrite:
        return filepath
    else:
        filepath.unlink(missing_ok=True)

    if filepath.exists():
        return filepath

    urlretrieve(url, filepath, MyProgressBar())
    return filepath


def unzip(filepath, remove_zip=True):
    print(f"Extracting: {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(f"Could not find file: {filepath}")

    filepath = Path(filepath)
    dirpath = filepath.parent

    with zipfile.ZipFile(filepath) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(dirpath)

    if remove_zip:
        filepath.unlink(missing_ok=True)

    return Path(dirpath).joinpath(zip_dir)


def download_zip(url, dirpath, overwrite=False, remove_zip=True):
    dirpath = Path(dirpath)

    if dirpath.exists() and not overwrite:
        print(f'Found {dirpath} - skip')
        return

    if dirpath.exists() and overwrite:
        shutil.rmtree(dirpath)

    filepath = download(url, dirpath.parent, overwrite=overwrite)
    zip_dir = unzip(filepath, remove_zip=remove_zip)
    shutil.move(zip_dir, dirpath)


def download_tagger(dirpath):
    url = 'http://nlp.stanford.edu/software/stanford-postagger-2015-01-29.zip'
    tagger_dir = 'stanford-tagger'

    download_zip(url, Path(dirpath).joinpath(tagger_dir))


def download_parser(dirpath):
    url = 'http://nlp.stanford.edu/software/stanford-parser-full-2015-01-29.zip'
    tagger_dir = 'stanford-parser'

    download_zip(url, Path(dirpath).joinpath(tagger_dir))


def download_wordvecs(dirpath):
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    dirpath = Path(dirpath)

    if dirpath.exists():
        print('Found Glove vectors - skip')
        return
    else:
        dirpath.mkdir(parents=True)

    unzip(download(url, dirpath))


def download_sst(dirpath):
    url = 'http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip'
    dirpath = Path(dirpath)

    if dirpath.exists():
        print('Found SST dataset - skip')
        return

    unzip(download(url, dirpath.parent))
    # rename dir from 'stanfordSentimentTreebank' to 'sst'
    dirpath.parent.joinpath('stanfordSentimentTreebank').rename(dirpath.parent.joinpath('sst'))
    shutil.rmtree(dirpath.parent.joinpath('__MACOSX'))  # remove extraneous dir


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # data
    data_dir = os.path.join(base_dir, 'data')
    wordvec_dir = os.path.join(data_dir, 'glove')
    sst_dir = os.path.join(data_dir, 'sst')

    # libraries
    lib_dir = os.path.join(base_dir, 'lib')

    # download dependencies
    download_tagger(lib_dir)
    download_parser(lib_dir)
    download_wordvecs(wordvec_dir)
    download_sst(sst_dir)
