import torch
import clip
import faiss
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Step 1: Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Step 2: Load the precomputed image embeddings and FAISS index
# Assume `image_features` is an array of CLIP features for all frames, and `index` is the FAISS index
image_features = np.load("./data/clip-features-vit-b32-sample/L01_V008.npy")  # Replace with your actual file path

#=== Index FAISS ===

# Get the dimensionality of your features
d = image_features.shape[1]  # For example, 512 if using CLIP with ViT-B/32

# Create a FAISS index for L2 distance (exact search)
index = faiss.IndexFlatL2(d)

# If you want to use cosine similarity instead of L2, you can normalize your features first
# image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)

# Step 4: Add CLIP features to the FAISS index
index.add(image_features)

# Optionally, save the index to disk for later use
faiss.write_index(index, "faiss_index.index")
print("Index saved to disk")
#==============

index = faiss.read_index("faiss_index.index")  # Replace with your actual index file path

# Step 3: Convert text query to an embedding using CLIP
def text_to_embedding(text):
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

# Step 4: Perform a search in the FAISS index using the text embedding
def search_frames(query, top_k=5):
    text_embedding = text_to_embedding(query)
    D, I = index.search(text_embedding, top_k)  # `D` is distances, `I` is indices of nearest neighbors
    return I

# Step 5: Retrieve and display results (this part is simplified)
# Assume `frame_paths` is a list of file paths corresponding to the frames in `image_features`
# frame_paths = np.load("frame_paths.npy")  # Replace with your actual file path

# Sample search query
query = "santa claus waving"

# Search for the top 5 frames that match the description
result_indices = search_frames(query, top_k=5)

# Print the file paths of the matching frames
for idx in result_indices[0]:
    frame = idx + 1
    print(frame)

print("Finished searching")
