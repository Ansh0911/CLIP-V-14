# Image Similarity Search using CLIP and FAISS

This project demonstrates how to find similar images from a folder using CLIP (Contrastive Language-Image Pre-Training) model for feature extraction and FAISS (Facebook AI Similarity Search) for efficient nearest neighbor search. The CLIP model encodes images into vector embeddings, and FAISS indexes these vectors to facilitate fast similarity searches based on cosine similarity.

## Project Setup

### Prerequisites

Make sure you have the following dependencies installed:
- `torch`
- `clip-by-openai`
- `faiss-cpu` (or `faiss-gpu` if you're using a GPU)
- `Pillow`
- `opencv-python`

You can install them using `pip`:

```bash
pip install torch clip-by-openai faiss-cpu Pillow opencv-python
```

### Project Structure

```plaintext
.
├── BACKUP MARBLE/           # Folder containing images to index
│   ├── image1.jpg
│   └── image2.png
├── FIND THIS MARBLE/        # Folder containing query image for similarity search
│   └── query_image.jpg
├── index_images.py          # Python script that performs indexing and search
├── README.md                # This README file
└── requirements.txt         # Optional file to list project dependencies
```

### How it Works

1. **Feature Extraction using CLIP**: The CLIP model from OpenAI is used to extract features from images. CLIP can process images and text in a unified space, making it great for tasks like finding similar images.

2. **Image Indexing with FAISS**: FAISS is used to efficiently store and search through high-dimensional image feature vectors. It supports various distance metrics, but here, we use cosine similarity for comparison.

3. **Similarity Search**: Given a query image, the program computes its feature vector using the CLIP model, and then FAISS is used to search the most similar images from the indexed dataset.

### How to Use

1. Place the images you want to index inside the `BACKUP MARBLE/` folder.
2. Place your query image inside the `FIND THIS MARBLE/` folder.
3. Run the script `index_images.py`:

```bash
python index_images.py
```

### How It Works:
- The script loads all images from the `BACKUP MARBLE/` folder and extracts their feature vectors using the CLIP model.
- These feature vectors are then indexed in FAISS using cosine similarity.
- You can then search for similar images by providing a query image from the `FIND THIS MARBLE/` folder.

### Example Output:

When you run the script, it will output the top `K` similar images (default `K=3`) to the query image, along with their similarity percentage.

```plaintext
Indexed 100 images.
Top Similar Images: [('image123.jpg', 97.56), ('image456.png', 95.43), ('image789.jpg', 93.11)]
```

### Adjusting Parameters

- **Top K Results**: You can change the number of top similar images by adjusting the `top_k` parameter in the `find_similar_images` function.
  
  Example:
  ```python
  similar_images = find_similar_images(query_image, top_k=5)
  ```

- **Image Folder Paths**: Make sure the `image_folder` and `query_image` variables point to the correct paths where your images are stored.

### Notes

- The code assumes that the images in the `BACKUP MARBLE/` folder are in `.jpg` or `.png` formats.
- The CLIP model is loaded onto the GPU if available; otherwise, it will fall back to using the CPU.
  
### Troubleshooting

1. **CLIP Model Loading Error**: Ensure that you have `torch` installed and CUDA is available if you're using the GPU.
2. **FAISS Error**: If you encounter any issues with FAISS, ensure that you have the correct version installed (`faiss-cpu` for CPU or `faiss-gpu` for GPU).

