# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Input and output files
# INPUT_FILE = "D:/Infosys Springboard Virtual Internship 6.0/Internal-Chatbot-with-RBAC/Chunking/all_chunks.json"
# OUTPUT_EMBEDDINGS = "chunk_embeddings.npy"
# OUTPUT_INDEX = "embedding_index.json"

# # Load the model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load chunks
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     chunks = json.load(f)

# # Extract text for embeddings
# contents = [chunk["chunk_text"] for chunk in chunks]  # updated key

# # Create embeddings
# embeddings = model.encode(contents, show_progress_bar=True)

# # Save embeddings as numpy file
# np.save(OUTPUT_EMBEDDINGS, embeddings)

# # Save index mapping chunk_id -> embedding index
# index = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
# with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
#     json.dump(index, f, ensure_ascii=False, indent=4)

# print("Embeddings and index saved successfully!")

# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import os

# # Paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# INPUT_FILE = os.path.join(BASE_DIR, "chunking", "student_chunks.json")
# OUTPUT_EMBEDDINGS = os.path.join(BASE_DIR, "Embeddings", "chunk_embeddings.npy")
# OUTPUT_INDEX = os.path.join(BASE_DIR, "Embeddings", "embedding_index.json")

# # Load model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Load chunks
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     chunks = json.load(f)

# # Extract text for embeddings
# contents = [chunk["chunk_text"] for chunk in chunks]  # use chunk_text key

# # Create embeddings
# embeddings = model.encode(contents, show_progress_bar=True)

# # Save embeddings
# np.save(OUTPUT_EMBEDDINGS, embeddings)

# # Save index mapping chunk_id -> embedding index
# index = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
# with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
#     json.dump(index, f, ensure_ascii=False, indent=4)

# print("Embeddings and index saved successfully!")


import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "chunking", "student_chunks.json")
OUTPUT_EMBEDDINGS = os.path.join(BASE_DIR, "Embeddings", "chunk_embeddings.npy")
OUTPUT_INDEX = os.path.join(BASE_DIR, "Embeddings", "embedding_index.json")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load chunks
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract text for embeddings
contents = [chunk["chunk_text"] for chunk in chunks]

# Create embeddings
embeddings = model.encode(contents, show_progress_bar=True)

# Save embeddings
np.save(OUTPUT_EMBEDDINGS, embeddings)

# Create index in your required format:
# ENGINEERING_CHUNK_1, FINANCE_CHUNK_13, etc.
index_list = [f"{chunk['chunk_id'].split('_')[0]}_CHUNK_{i+1}" for i, chunk in enumerate(chunks)]

# Save as LIST
with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
    json.dump(index_list, f, ensure_ascii=False, indent=4)

print("Embeddings and index saved successfully!")




