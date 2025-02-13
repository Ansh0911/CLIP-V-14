import torch
import clip
import faiss
import os
import numpy as np
from PIL import Image
import cv2

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Here we will load the clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)


# This function will extract CLIP features from the image
def extract_features(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy().flatten()  # Convert to NumPy array


# Load Images from Folder 
image_folder = "BACKUP MARBLE/"  
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# Extract Features 
image_vectors = np.array([extract_features(os.path.join(image_folder, img)) for img in image_files])

# Normalize vectors to unit length for cosine similarity
image_vectors = image_vectors / np.linalg.norm(image_vectors, axis=1, keepdims=True)


# Create FAISS Index
dimension = image_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # L2 Distance or Euclidian Distance for similarity search
index.add(image_vectors)

print(f"Indexed {len(image_vectors)} images.")


# # Calculate the max and min distances (for normalization)
# distances, _ = index.search(image_vectors, 2)  # Get distances of all images to themselves
# max_distance = np.max(distances)
# min_distance = np.min(distances)




# Function to Find Similar Images (CLIP)
def find_similar_images(query_image_path, top_k=3):
    query_vector = extract_features(query_image_path).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)  # Get the top K ranked results
    query_vector = query_vector / np.linalg.norm(query_vector)
   
    similarities, indices = index.search(query_vector, top_k)

    similarity_percentages = [round(sim * 100, 2) for sim in similarities[0]]
    
    similar_images_with_percentage = [(image_files[i], similarity_percentages[idx]) for idx, i in enumerate(indices[0])]

    return similar_images_with_percentage



# Example: Search Similar Images
query_image = "FIND THIS MARBLE/BEIGE20.jpg"  
similar_images = find_similar_images(query_image)

print("Top Similar Images:", similar_images)