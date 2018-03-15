from pathlib import Path


file = 'face.zip'
file = 'iris.zip'
file = 'ear.zip'

# DOWNLOAD
dl_dir = 'cache/raw/'
if Path(dl_dir + file).is_file():
    print('file ' + dl_dir + file + ' already present')
else:
    print('downloading')

    import requests

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
if Path(extract_dir + file).is_file():
    print('file ' + extract_dir + file + ' already present')
else:
    print('unzipping')
    import zipfile

    zip_ref = zipfile.ZipFile(dl_dir + file, 'r')
    zip_ref.extractall(extract_dir + file)
    zip_ref.close()
