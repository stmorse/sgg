import os
import tarfile
import urllib.request

BASE_PATH = '/sciclone/geograd/stmorse/reddit/snap'

# url = "https://snap.stanford.edu/data/reddit_chain_networks.tar.gz"
# dataset_path = os.path.join(BASE_PATH, 'reddit_chain_networks.tar.gz')
# extract_path = os.path.join(BASE_PATH, 'reddit_chain_networks')

# # Download the dataset
# print("Downloading chain networks ...")
# urllib.request.urlretrieve(url, filename=dataset_path)
# print("Download complete.")

# # Extract the tarball
# print("Extracting chain networks ...")
# with tarfile.open(dataset_path, 'r:gz') as tar:
#     tar.extractall(path=extract_path)
# print("Extraction complete.")

#

url = "https://snap.stanford.edu/data/reddit_reply_networks.tar.gz"
dataset_path = os.path.join(BASE_PATH, 'reddit_reply_networks.tar.gz')
extract_path = os.path.join(BASE_PATH, 'reddit_reply_networks')

# Download the dataset
print("Downloading reply networks ...")
urllib.request.urlretrieve(url, filename=dataset_path)
print("Download complete.")

# Extract the tarball
print("Extracting reply networks ...")
with tarfile.open(dataset_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)
print("Extraction complete.")