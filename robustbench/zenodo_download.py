import hashlib
import shutil
from pathlib import Path
from typing import Set
import wget
import requests
import math

ZENODO_ENTRY_POINT = "https://zenodo.org/api"
RECORDS_ENTRY_POINT = f"{ZENODO_ENTRY_POINT}/records/"

CHUNK_SIZE = 65536

class DownloadError(Exception):
    pass

def bar_custom(current, total, width=80):
    width=30
    avail_dots = width-2
    shaded_dots = int(math.floor(float(current) / total * avail_dots))
    percent_bar = '[' + '■'*shaded_dots + ' '*(avail_dots-shaded_dots) + ']'
    progress = "%d%% %s [%d / %d]" % (current / total * 100, percent_bar, current, total)
    return progress

def download_file(url: str, filekey: str, save_dir: Path) -> Path:
    local_filename = save_dir / filekey
    wget.download(url, out=str(local_filename), bar=bar_custom)
    return local_filename

def file_md5(filename: Path) -> str:
    """Computes the MD5 hash of a given file"""
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(32768), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def zenodo_download(record_id: str, filenames_to_download: Set[str], save_dir: Path) -> None:
    """Downloads the given files from the given Zenodo record.

    :param record_id: The ID of the record.
    :param filenames_to_download: The files to download from the record.
    :param save_dir: The directory where the files should be saved.
    """
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    url = f"{RECORDS_ENTRY_POINT}/{record_id}"
    res = requests.get(url)
    files = res.json()["files"]
    files_to_download = list(
        filter(lambda file: file["key"] in filenames_to_download, files))

    for file in files_to_download:
        if (save_dir / file["key"]).exists():
            continue
        file_url = file["links"]["self"]
        file_checksum = file["checksum"].split(":")[-1]
        file_key = file["key"]
        filename = download_file(file_url, file_key, save_dir)
        if file_md5(filename) != file_checksum:
            raise DownloadError(
                "The hash of the downloaded file does not match"
                " the expected one.")
        print("Download finished, extracting...")
        shutil.unpack_archive(filename, extract_dir=save_dir)
        print("Downloaded and extracted.")