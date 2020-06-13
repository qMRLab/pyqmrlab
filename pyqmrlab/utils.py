import cgi
import zipfile
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util import Retry
import tempfile
from pathlib import Path
from tqdm import tqdm


def download_data(url_data, folder=None):
    """ Downloads and extracts zip files from the web.
    :return: 0 - Success, 1 - Encountered an exception.
    """
    # Download
    try:
        print("Trying URL: %s" % url_data)
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 503, 504])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        response = session.get(url_data, stream=True)

        if "Content-Disposition" in response.headers:
            _, content = cgi.parse_header(response.headers["Content-Disposition"])
            zip_filename = content["filename"]
        else:
            print("Unexpected: link doesn't provide a filename")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / zip_filename
            with open(tmp_path, "wb") as tmp_file:
                total = int(response.headers.get("content-length", 1))
                tqdm_bar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                    ascii=True,
                )
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        dl_chunk = len(chunk)
                        tqdm_bar.update(dl_chunk)
                tqdm_bar.close()
            # Unzip
            print("Unzip...")
            try:
                zf = zipfile.ZipFile(str(tmp_path))

                if folder == None:
                    folder = Path.cwd()
                else:
                    folder = Path(folder)
                print(folder / Path(zip_filename).stem)
                zf.extractall(str(folder / Path(zip_filename).stem))
            except (zipfile.BadZipfile):
                print("ERROR: ZIP package corrupted. Please try downloading again.")
                return 1
            print("--> Folder created: " + str(folder / Path(zip_filename).stem))
    except Exception as e:
        print("ERROR: %s" % e)
        return 1
    return 0
