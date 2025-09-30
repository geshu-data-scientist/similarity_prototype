import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
import streamlit as st
import cv2
import imghdr
import sqlite3
from pathlib import Path
import shutil

# --- Page Configuration ---
st.set_page_config(
    page_title="Image Similarity Engine",
    page_icon="🖼️",
    layout="wide"
)

# --- Configuration & Setup ---
# All persistent data will be stored here. This works locally.
# On Streamlit Community Cloud, this will be an ephemeral filesystem.
DATA_DIR = "data"
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
REPO_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, 'repo')
NEW_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, 'new')
DATABASE_FILE = os.path.join(DATA_DIR, "similarity_results.db")
SUPPORTED_TYPES = ['jpeg', 'png']
SIMILARITY_THRESHOLD = 0.99

# Ensure all necessary folders exist
for folder in [UPLOAD_FOLDER, REPO_EMBEDDINGS_DIR, NEW_EMBEDDINGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_model():
    """Loads the ResNet model and caches it for the app session."""
    st.write("Cache miss: Loading ResNet model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # This downloads and loads the official pre-trained weights automatically
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(resnet.children())[:-1])
    model.to(device)
    model.eval()
    st.write("Model loaded successfully.")
    return model, device

MODEL, DEVICE = load_model()

TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- All Helper & Logic Functions ---
def is_unicolor(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None: return False
        return np.all(image == image[0, 0])
    except Exception: return False

def is_black_or_white(image_array, threshold=0.05):
    black = np.sum(image_array <= 255 * threshold) / image_array.size > 0.9
    white = np.sum(image_array >= 255 * (1 - threshold)) / image_array.size > 0.9
    if black: return 'black'
    if white: return 'white'
    return None

def crop_top_bottom(image):
    h, w = image.shape[:2]
    per = int(h / 10)
    return image[per:h - per, :]

def count_edges(image, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sum(cv2.Canny(gray, low_thresh, high_thresh) > 0)

def process_and_validate_image(image_path):
    """Runs all checks and returns an embedding or a failure reason string."""
    try:
        if imghdr.what(image_path) not in SUPPORTED_TYPES:
            return None, f"Unsupported file type ({imghdr.what(image_path) or 'unknown'})"
        if is_unicolor(image_path):
            return None, "Image is unicolor."
        image_cv = cv2.imread(image_path)
        if image_cv is None: return None, "Could not read image."
        if count_edges(crop_top_bottom(image_cv)) < 100:
            return None, "Low edge count, likely blurry or blank."
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        bw_result = is_black_or_white(np.array(image_pil.convert('L')))
        if bw_result: return None, f"Image is predominantly {bw_result}."
        
        image_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = MODEL(image_tensor).squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        return (embedding / norm if norm > 0 else embedding), "Success"
    except Exception as e:
        return None, f"An error occurred: {e}"

def load_embeddings_from_dir(directory):
    keys, embs_list = [], []
    for f in Path(directory).glob('*.npy'):
        try:
            data = np.load(f, allow_pickle=True).item()
            keys.extend(data.keys())
            embs_list.append(list(data.values())[0])
        except Exception: pass
    return keys, np.array(embs_list)

def find_similar_pairs_vectorized(new_embs, repo_embs, device):
    with torch.no_grad():
        new_tensor = torch.nn.functional.normalize(torch.tensor(new_embs, dtype=torch.float32, device=device))
        repo_tensor = torch.nn.functional.normalize(torch.tensor(repo_embs, dtype=torch.float32, device=device))
        sim_matrix = torch.matmul(new_tensor, repo_tensor.T)
        indices = torch.where(sim_matrix >= SIMILARITY_THRESHOLD)
        return indices[0].cpu().numpy(), indices[1].cpu().numpy(), sim_matrix[indices].cpu().numpy()

def setup_database():
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS similarity_pairs (
            id INTEGER PRIMARY KEY, new_image_path TEXT, repo_image_path TEXT,
            similarity_score REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()

def insert_into_db(results):
    if not results: return
    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.executemany("INSERT INTO similarity_pairs (new_image_path, repo_image_path, similarity_score) VALUES (?, ?, ?)", results)
        conn.commit()

def get_recent_results(limit=10):
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.execute("SELECT new_image_path, repo_image_path, similarity_score FROM similarity_pairs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cursor.fetchall()

# --- Streamlit UI ---
st.title("🖼️ Image Similarity Engine")
st.markdown("Upload new images, compare them against a repository, and find visual duplicates. **Note:** *The filesystem on Streamlit Cloud is ephemeral, so the repository will reset after the app sleeps.*")

# Ensure database is ready
setup_database()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Actions")
    
    # --- 1. Upload Image ---
    uploaded_file = st.file_uploader("1. Upload an Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            filename = uploaded_file.name
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            embedding, msg = process_and_validate_image(upload_path)
            
            if embedding is not None:
                embedding_data = {upload_path: embedding}
                save_path = os.path.join(NEW_EMBEDDINGS_DIR, Path(filename).stem + '.npy')
                np.save(save_path, embedding_data)
                st.success(f"'{filename}' passed validation and its embedding was saved.")
            else:
                st.error(f"'{filename}' failed validation: {msg}")

    # --- 2. Run Comparison ---
    st.markdown("---")
    if st.button("2. Find Duplicates & Update Repository", type="primary"):
        with st.spinner("Running comparison..."):
            new_keys, new_embs = load_embeddings_from_dir(NEW_EMBEDDINGS_DIR)
            repo_keys, repo_embs = load_embeddings_from_dir(REPO_EMBEDDINGS_DIR)

            if not new_keys:
                st.warning('No new images to compare.')
            else:
                found_pairs_count = 0
                if repo_keys:
                    new_indices, repo_indices, scores = find_similar_pairs_vectorized(new_embs, repo_embs, DEVICE)
                    results_to_insert = [
                        (new_keys[n_idx], repo_keys[r_idx], float(score))
                        for n_idx, r_idx, score in zip(new_indices, repo_indices, scores)
                    ]
                    insert_into_db(results_to_insert)
                    found_pairs_count = len(results_to_insert)
                
                # Merge new embeddings into the repository
                for f in os.listdir(NEW_EMBEDDINGS_DIR):
                    shutil.move(os.path.join(NEW_EMBEDDINGS_DIR, f), os.path.join(REPO_EMBEDDINGS_DIR, f))
                
                st.success(f'Comparison complete. Found {found_pairs_count} similar pairs. New images have been moved to the repository.')

with col2:
    st.header("System Status")
    new_files_count = len(os.listdir(NEW_EMBEDDINGS_DIR))
    repo_files_count = len(os.listdir(REPO_EMBEDDINGS_DIR))
    st.info(f"**{new_files_count}** new images are ready for comparison.")
    st.info(f"**{repo_files_count}** images are in the main repository.")

    st.header("Recent Duplicates Found")
    recent_results = get_recent_results(10)
    if recent_results:
        st.dataframe(recent_results, use_container_width=True, column_config={
            0: "New Image", 1: "Repository Image", 2: "Score"
        })
    else:
        st.write("No recent duplicates found in the database.")