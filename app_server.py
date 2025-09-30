import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import cv2
import imghdr
import sqlite3
import logging
from pathlib import Path
from math import ceil
import shutil

#uncomment for render
# --- Configuration ---
# All persistent data will be stored here.
DATA_DIR = "/var/data/similarity_app"
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
REPO_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, 'repo')
NEW_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, 'new')
DATABASE_FILE = os.path.join(DATA_DIR, "similarity_results.db")

#coment for render
'''# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_DIR = 'embeddings'
REPO_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, 'repo')
NEW_EMBEDDINGS_DIR = os.path.join(EMBEDDINGS_DIR, 'new')
DATABASE_FILE = "similarity_results.db"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SUPPORTED_TYPES = ['jpeg', 'png']
SIMILARITY_THRESHOLD = 0.99'''

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-super-secret-key' # Needed for flashing messages

# --- Initial Setup ---
# Ensure all necessary folders exist
for folder in [UPLOAD_FOLDER, REPO_EMBEDDINGS_DIR, NEW_EMBEDDINGS_DIR]:
    os.makedirs(folder, exist_ok=True)

# Set up logging
log_file = "local_similarity.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])

# --- Model Loading (Once at Startup) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")
local_weights_path = "resnet18-f37072fd.pth"
if not os.path.exists(local_weights_path):
    logging.error("ResNet weights file 'resnet18-f37072fd.pth' not found!")
    exit() # Or handle this more gracefully

'''resnet = models.resnet18(weights=None)
resnet.load_state_dict(torch.load(local_weights_path, map_location=DEVICE))
MODEL = nn.Sequential(*list(resnet.children())[:-1])
MODEL.to(DEVICE)
MODEL.eval()'''

 --- Model Loading (Once at Startup) ---
logging.info(f"Using device: {DEVICE}")

# This single line downloads and loads the official pre-trained weights automatically
from torchvision.models import ResNet18_Weights
resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

MODEL = nn.Sequential(*list(resnet.children())[:-1])
MODEL.to(DEVICE)
MODEL.eval()
#uncomment for github use

TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Validation and Helper Functions ---
def is_unicolor(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None: return False
        return np.all(image == image[0, 0])
    except Exception: return False

def is_black_or_white(image_array, threshold=0.05):
    black_pixels = np.sum(image_array <= 255 * threshold) / image_array.size
    white_pixels = np.sum(image_array >= 255 * (1 - threshold)) / image_array.size
    if black_pixels > 0.9: return 'black'
    if white_pixels > 0.9: return 'white'
    return None

def crop_top_bottom(image):
    h, w = image.shape[:2]
    per = int(h / 10)
    return image[per:h - per, :]

def count_edges(image, low_thresh=50, high_thresh=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return np.sum(edges > 0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Core Logic for Processing and Comparison ---
def process_and_validate_image(image_path):
    """Runs all checks and returns an embedding or a failure reason string."""
    try:
        if imghdr.what(image_path) not in SUPPORTED_TYPES:
            return None, f"Unsupported file type ({imghdr.what(image_path) or 'unknown'})"
        if is_unicolor(image_path):
            return None, "Image is unicolor."
        
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            return None, "Could not read image with OpenCV."
        
        if count_edges(crop_top_bottom(image_cv)) < 100:
            return None, "Low edge count, likely blurry or blank."
        
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        image_array_bw = np.array(image_pil.convert('L'))
        bw_result = is_black_or_white(image_array_bw)
        if bw_result:
            return None, f"Image is predominantly {bw_result}."
        
        image_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = MODEL(image_tensor).squeeze().cpu().numpy()
        
        norm = np.linalg.norm(embedding)
        return (embedding / norm if norm > 0 else embedding), "Success"
        
    except Exception as e:
        return None, f"An error occurred: {e}"

def load_embeddings_from_dir(directory):
    """Loads all embeddings from a directory into a key list and a numpy array."""
    keys, embs_list = [], []
    for f in Path(directory).glob('*.npy'):
        try:
            data = np.load(f, allow_pickle=True).item()
            keys.extend(data.keys())
            embs_list.append(list(data.values())[0])
        except Exception as e:
            logging.warning(f"Could not load {f}: {e}")
    return keys, np.array(embs_list)

def find_similar_pairs_vectorized(new_embs, repo_embs, device):
    """Vectorized similarity search."""
    with torch.no_grad():
        new_tensor = torch.nn.functional.normalize(torch.tensor(new_embs, dtype=torch.float32, device=device))
        repo_tensor = torch.nn.functional.normalize(torch.tensor(repo_embs, dtype=torch.float32, device=device))
        sim_matrix = torch.matmul(new_tensor, repo_tensor.T)
        indices = torch.where(sim_matrix >= SIMILARITY_THRESHOLD)
        return indices[0].cpu().numpy(), indices[1].cpu().numpy(), sim_matrix[indices].cpu().numpy()

# --- Database Functions ---
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

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Renders the main page with status info."""
    new_files_count = len(os.listdir(NEW_EMBEDDINGS_DIR))
    repo_files_count = len(os.listdir(REPO_EMBEDDINGS_DIR))
    recent_results = get_recent_results()
    return render_template('index.html', new_count=new_files_count, repo_count=repo_files_count, results=recent_results)

@app.route('/upload', methods=['POST'])
def handle_upload():
    """Handles image upload, validation, and embedding generation."""
    if 'file' not in request.files or not request.files['file'].filename:
        flash('No file selected.', 'warning')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        embedding, msg = process_and_validate_image(upload_path)
        
        if embedding is not None:
            embedding_data = {upload_path: embedding}
            save_path = os.path.join(NEW_EMBEDDINGS_DIR, Path(filename).stem + '.npy')
            np.save(save_path, embedding_data)
            flash(f"'{filename}' passed validation and embedding was saved.", 'success')
        else:
            flash(f"'{filename}' failed validation: {msg}", 'danger')
            
    return redirect(url_for('index'))

@app.route('/compare', methods=['POST'])
def handle_compare():
    """Runs the comparison and merges new embeddings into the repo."""
    new_keys, new_embs = load_embeddings_from_dir(NEW_EMBEDDINGS_DIR)
    repo_keys, repo_embs = load_embeddings_from_dir(REPO_EMBEDDINGS_DIR)
    
    if not new_keys:
        flash('No new images to compare.', 'warning')
        return redirect(url_for('index'))
    
    if repo_keys:
        new_indices, repo_indices, scores = find_similar_pairs_vectorized(new_embs, repo_embs, DEVICE)
        
        results_to_insert = [
            (new_keys[n_idx], repo_keys[r_idx], float(score))
            for n_idx, r_idx, score in zip(new_indices, repo_indices, scores)
        ]
        insert_into_db(results_to_insert)
        flash(f'Comparison complete. Found {len(results_to_insert)} similar pairs.', 'info')
    else:
        flash('Repository is empty. New images will form the initial repository.', 'info')

    # Merge new embeddings into the repository
    #for f in os.listdir(NEW_EMBEDDINGS_DIR):
     #   os.rename(os.path.join(NEW_EMBEDDINGS_DIR, f), os.path.join(REPO_EMBEDDINGS_DIR, f))
    for f in os.listdir(NEW_EMBEDDINGS_DIR):
        source_path = os.path.join(NEW_EMBEDDINGS_DIR, f)
        destination_path = os.path.join(REPO_EMBEDDINGS_DIR, f)
        shutil.move(source_path, destination_path)
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    setup_database()
    app.run(debug=False, host='0.0.0.0')