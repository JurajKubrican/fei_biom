import os.path

# file = 'face.zip'
# file = 'iris.zip'
file = 'ear.zip'

# DOWNLOAD
dl_dir = 'cache/raw/'
if (os.path.isfile(dl_dir + file)):

    import requests
    from pathlib import Path

    url = 'http://147.175.106.116:81/Biometria/'
    r = requests.get(url + file, stream=True)

    path = Path(dl_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(dl_dir + file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# UNZIP
extract_dir = 'cache/extract/'
if (os.path.isfile(extract_dir + file)):
    import zipfile

    zip_ref = zipfile.ZipFile(dl_dir + file, 'r')
    zip_ref.extractall(extract_dir)
    zip_ref.close()
