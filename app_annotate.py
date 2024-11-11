from flask import Flask, render_template, jsonify, request, send_file
from pathlib import Path
import logging
import os
from datetime import datetime
import sys
import shutil

# Add project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize labeler at module level
def create_labeler():
    """Create and configure the labeler"""
    from llama_annotate import LlamaLabeler, cleanup_directories
    return LlamaLabeler(), cleanup_directories

# Create Flask app
app = Flask(__name__)

# Initialize labeler and get cleanup function
labeler, cleanup_directories = create_labeler()

def init_static():
    """Initialize static directory with images"""
    logger.info("Initializing static directory...")
    static_dir = Path('static/images')
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        # Create directories
        (static_dir / split).mkdir(parents=True, exist_ok=True)
        (static_dir / 'viz' / split).mkdir(parents=True, exist_ok=True)
        
        # Copy original images
        src_img_dir = labeler.images_dir / split
        if src_img_dir.exists():
            for img in src_img_dir.glob('*.[jp][pn][g]'):
                dst = static_dir / split / img.name
                try:
                    if not dst.exists() or not dst.samefile(img):
                        shutil.copy2(img, dst)
                        logger.debug(f"Copied {img.name} to {dst}")
                except OSError:
                    shutil.copy2(img, dst)
                    logger.debug(f"Copied {img.name} to {dst}")
        
        # Copy visualizations
        src_viz_dir = labeler.viz_dir / split
        if src_viz_dir.exists():
            for viz in src_viz_dir.glob('viz_*'):
                dst = static_dir / 'viz' / split / viz.name
                try:
                    if not dst.exists() or not dst.samefile(viz):
                        shutil.copy2(viz, dst)
                        logger.debug(f"Copied {viz.name} to {dst}")
                except OSError:
                    shutil.copy2(viz, dst)
                    logger.debug(f"Copied {viz.name} to {dst}")

    logger.info("Static directory initialized")

# Initialize static files after creating labeler
labeler, cleanup_directories = create_labeler()
init_static()

@app.route('/')
def index():
    """Main page showing dataset overview"""
    dataset_info = get_dataset_info()
    return render_template('index.html', dataset_info=dataset_info)

@app.route('/process', methods=['POST'])
def process_images():
    """Process all images in dataset"""
    try:
        # Clean up previous results
        cleanup_directories(labeler.dataset_root)
        
        # Process each split
        results = {}
        for split in ['train', 'val', 'test']:
            split_dir = labeler.images_dir / split
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(split_dir.glob(ext)))
            
            results[split] = {
                'total': len(image_files),
                'processed': 0,
                'success': 0
            }
            
            # Process images in batches
            batch_size = 5  # Process 5 images at a time
            for i in range(0, len(image_files), batch_size):
                batch = image_files[i:i + batch_size]
                
                # Process batch
                for img_path in batch:
                    results[split]['processed'] += 1
                    if labeler.process_image(img_path):
                        results[split]['success'] += 1
                
                # Copy files to static after each batch
                logger.info("Copying files to static directory...")
                init_static()
                    
        return jsonify({
            'status': 'success',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/status')
def get_status():
    """Get current processing status"""
    dataset_info = get_dataset_info()
    return jsonify(dataset_info)

@app.route('/visualization/<split>/<filename>')
def get_visualization(split, filename):
    """Serve visualization images"""
    # Remove 'viz_' prefix if it's already in the filename
    if not filename.startswith('viz_'):
        viz_filename = f"viz_{filename}"
    else:
        viz_filename = filename
        
    viz_path = labeler.viz_dir / split / viz_filename
    logger.debug(f"Looking for visualization at: {viz_path}")
    
    if viz_path.exists():
        return send_file(str(viz_path.absolute()))
    logger.warning(f"Visualization not found: {viz_path}")
    return "Image not found", 404

@app.route('/static/images/<split>/<filename>')
def serve_image(split, filename):
    """Serve images from dataset directory"""
    image_path = labeler.images_dir / split / filename
    if image_path.exists():
        return send_file(image_path)
    return "Image not found", 404

def get_dataset_info():
    """Get information about the dataset"""
    info = {}
    for split in ['train', 'val', 'test']:
        split_info = {
            'images': [],
            'total_images': 0,
            'processed_images': 0,
            'visualizations': 0
        }
        
        # Get original images
        image_dir = labeler.images_dir / split
        images = []
        if image_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images.extend(list(image_dir.glob(ext)))
        
        split_info['total_images'] = len(images)
        
        # Get ALL images with their visualizations (removed the [:5] limit)
        for img_path in sorted(images):
            viz_path = labeler.viz_dir / split / f"viz_{img_path.name}"
            label_path = labeler.labels_dir / split / f"{img_path.stem}.txt"
            
            # Also check static directory for visualizations
            static_viz_path = Path('static/images/viz') / split / f"viz_{img_path.name}"
            has_viz = viz_path.exists() or static_viz_path.exists()
            
            split_info['images'].append({
                'name': img_path.name,
                'has_visualization': has_viz,
                'has_label': label_path.exists(),
                'path': f"{split}/{img_path.name}"
            })
        
        # Count processed images
        labels_dir = labeler.labels_dir / split
        if labels_dir.exists():
            split_info['processed_images'] = len(list(labels_dir.glob('*.txt')))
        
        # Count visualizations from both directories
        viz_dir = labeler.viz_dir / split
        static_viz_dir = Path('static/images/viz') / split
        viz_count = 0
        
        if viz_dir.exists():
            viz_count += len(list(viz_dir.glob('viz_*')))
        if static_viz_dir.exists():
            viz_count += len(list(static_viz_dir.glob('viz_*')))
            
        split_info['visualizations'] = viz_count
        
        info[split] = split_info
    
    return info

@app.route('/debug/paths')
def debug_paths():
    """Debug endpoint to check paths"""
    paths = {
        'root': str(labeler.dataset_root),
        'images': str(labeler.images_dir),
        'labels': str(labeler.labels_dir),
        'visualizations': str(labeler.viz_dir),
        'splits': {}
    }
    
    for split in ['train', 'val', 'test']:
        split_dir = labeler.images_dir / split
        if split_dir.exists():
            images = list(split_dir.glob('*.[jp][pn][g]'))
            paths['splits'][split] = {
                'dir': str(split_dir),
                'image_count': len(images),
                'sample_images': [img.name for img in images[:3]]
            }
    
    return jsonify(paths)

if __name__ == '__main__':
    # Initialize Flask app with a different port
    logger.info(f"Starting Llama Fire Annotator on http://127.0.0.1:5003")
    app.run(
        host='127.0.0.1',
        port=5003,
        debug=True, 
        use_reloader=False
    )