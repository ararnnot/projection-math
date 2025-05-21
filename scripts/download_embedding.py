import os
import zipfile
import urllib.request
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Run just once
# This script downloads the GloVe embeddings, extracts and converts to word2vec binary format
# Generates files resources/glove.6B.50d.kv and resources/glove.6B.50d.kv.vectors.npy

# Paths
glove_dir = "resources"
glove_zip_url = "http://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = os.path.join(glove_dir, "glove.6B.zip")
glove_txt_path = os.path.join(glove_dir, "glove.6B.50d.txt")
w2v_output_path = os.path.join(glove_dir, "glove.6B.50d.w2v.txt")

os.makedirs(glove_dir, exist_ok=True)

print("Downloading GloVe embeddings...")
urllib.request.urlretrieve(glove_zip_url, glove_zip_path)
print("Download complete.")

print("Extracting glove.6B.50d.txt...")
with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
    zip_ref.extract("glove.6B.50d.txt", path=glove_dir)
print("Extraction complete.")

print("Converting to word2vec format...")
glove2word2vec(glove_txt_path, w2v_output_path)
print("Conversion complete.")

print("Loading embeddings...")
model = KeyedVectors.load_word2vec_format(w2v_output_path, binary=False)

print("Most similar to 'king':", model.most_similar("king", topn=5))

model.save(os.path.join(glove_dir, "glove.6B.50d.kv"))

files_to_delete = [
    os.path.join(glove_dir, "glove.6B.50d.txt"),
    os.path.join(glove_dir, "glove.6B.zip"),
    os.path.join(glove_dir, "glove.6B.50d.w2v.txt"),
]
for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)