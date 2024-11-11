from flask import Flask, render_template, jsonify, request, send_file, Response, stream_with_context, url_for
from pathlib import Path
import logging
import os
from datetime import datetime
import sys
import shutil
import json
import time
from werkzeug.middleware.proxy_fix import ProxyFix

# Add project root to Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',  # Simplified format
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from libraries
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Initialize labeler at module level
def create_labeler():
    """Create and configure the labeler"""
    from llama_annotate import LlamaLabeler, cleanup_directories
    
    # Only clean if directories don't exist
    labels_dir = Path('data/binary_fire_dataset/labels')
    viz_dir = Path('data/binary_fire_dataset/visualizations')
    
    if not labels_dir.exists() and not viz_dir.exists():
        cleanup_directories()  # Only cleanup if fresh start
        
    return LlamaLabeler(), cleanup_directories

# Create Flask app
app = Flask(__name__)

# Add proxy support for ngrok
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure for ngrok
app.config['PREFERRED_URL_SCHEME'] = 'https'

# Initialize labeler and get cleanup function
labeler, cleanup_directories = create_labeler()

# Add global variable to track processing status
processing_status = {
    'is_processing': False,
    'current_image': '',
    'processed': 0,
    'total': 0
}

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
                    if not dst.exists():
                        shutil.copy2(img, dst)
                        logger.debug(f"Copied {img.name} to {dst}")
                except OSError:
                    shutil.copy2(img, dst)
                    logger.debug(f"Copied {img.name} to {dst}")
        
        # Copy visualizations from both possible source directories
        src_viz_dirs = [
            labeler.viz_dir / split,  # Original viz directory
            Path('static/images/viz') / split  # Static viz directory
        ]
        
        for src_viz_dir in src_viz_dirs:
            if src_viz_dir.exists():
                for viz in src_viz_dir.glob('viz_*'):
                    dst = static_dir / 'viz' / split / viz.name
                    try:
                        if not dst.exists():
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

def generate_events():
    """Generate SSE events"""
    while True:
        # Get current status
        data = {
            'is_processing': processing_status['is_processing'],
            'current_image': processing_status['current_image'],
            'processed': processing_status['processed'],
            'total': processing_status['total']
        }
        
        # Send event
        yield f"data: {json.dumps(data)}\n\n"
        time.sleep(1)  # Update every second

@app.route('/stream')
def stream():
    """SSE endpoint"""
    return Response(
        stream_with_context(generate_events()),
        mimetype='text/event-stream'
    )

@app.route('/process', methods=['POST'])
def process_images():
    """Process all images in dataset"""
    try:
        # Check if all images are already processed
        dataset_info = get_dataset_info()
        if dataset_info['processing_status']['all_processed']:
            return jsonify({
                'status': 'skipped',
                'message': 'All images have already been processed'
            })

        global processing_status
        processing_status['is_processing'] = True
        
        # Calculate total images
        total_images = 0
        for split in ['train', 'val', 'test']:
            split_dir = labeler.images_dir / split
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                total_images += len(list(split_dir.glob(ext)))
        
        processing_status['total'] = total_images
        processing_status['processed'] = 0
        
        logger.info(f"🚀 Starting to process {total_images} images")
        
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
            
            # Process each image
            for img_path in image_files:
                processing_status['current_image'] = img_path.name
                processing_status['processed'] += 1
                
                progress = (processing_status['processed'] / total_images) * 100
                logger.info(f"📸 [{progress:.1f}%] Processing {img_path.name} ({processing_status['processed']}/{total_images})")
                
                if labeler.process_image(img_path):
                    results[split]['success'] += 1
                    logger.info(f"✅ Fire detected in {img_path.name}")
                else:
                    logger.info(f"❌ No fire in {img_path.name}")
                
                results[split]['processed'] += 1
                
                # Copy to static and update frontend
                init_static()
                logger.info(f"🔄 Updated frontend with {img_path.name}")
                
        logger.info(f"✨ Completed processing {total_images} images")
        processing_status['is_processing'] = False
        
        return jsonify({
            'status': 'success',
            'results': results,
            'progress': 100,
            'message': f"Processed {total_images} images"
        })
        
    except Exception as e:
        processing_status['is_processing'] = False
        logger.error(f"❌ Processing failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/progress')
def get_progress():
    """Get current processing progress"""
    try:
        total_images = 0
        processed_images = 0
        
        for split in ['train', 'val', 'test']:
            labels_dir = labeler.labels_dir / split
            viz_dir = labeler.viz_dir / split
            
            # Count total images
            split_dir = labeler.images_dir / split
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                total_images += len(list(split_dir.glob(ext)))
            
            # Count processed images
            if labels_dir.exists():
                processed_images += len(list(labels_dir.glob('*.txt')))
        
        progress = (processed_images / total_images * 100) if total_images > 0 else 0
        
        return jsonify({
            'total': total_images,
            'processed': processed_images,
            'progress': progress
        })
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return jsonify({
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
    
    # Check both possible locations for visualizations
    possible_paths = [
        labeler.viz_dir / split / viz_filename,
        Path('static/images/viz') / split / viz_filename
    ]
    
    for viz_path in possible_paths:
        if viz_path.exists():
            return send_file(str(viz_path.absolute()))
    
    logger.warning(f"Visualization not found: {viz_filename}")
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
    total_images_all_splits = 0
    total_processed_all_splits = 0
    
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
        total_images_all_splits += split_info['total_images']
        
        # Count processed images
        labels_dir = labeler.labels_dir / split
        if labels_dir.exists():
            split_info['processed_images'] = len(list(labels_dir.glob('*.txt')))
            total_processed_all_splits += split_info['processed_images']
        
        logger.info(f"Split {split}: {split_info['processed_images']}/{split_info['total_images']} images processed")
        
        # Get ALL images with their visualizations
        for img_path in sorted(images):
            # Check both possible locations for visualizations
            viz_paths = [
                labeler.viz_dir / split / f"viz_{img_path.name}",
                Path('static/images/viz') / split / f"viz_{img_path.name}"
            ]
            label_path = labeler.labels_dir / split / f"{img_path.stem}.txt"
            
            has_viz = any(path.exists() for path in viz_paths)
            has_label = label_path.exists()
            
            split_info['images'].append({
                'name': img_path.name,
                'has_visualization': has_viz,
                'has_label': has_label,
                'path': f"{split}/{img_path.name}"
            })
        
        # Count visualizations from both directories
        viz_dirs = [
            labeler.viz_dir / split,
            Path('static/images/viz') / split
        ]
        
        viz_count = sum(
            len(list(viz_dir.glob('viz_*')))
            for viz_dir in viz_dirs
            if viz_dir.exists()
        )
        
        split_info['visualizations'] = viz_count
        
        info[split] = split_info
    
    logger.info(f"Total processed: {total_processed_all_splits}/{total_images_all_splits}")
    logger.info(f"All processed: {total_processed_all_splits >= total_images_all_splits}")
    
    # Add processing status information
    info['processing_status'] = {
        'all_processed': total_processed_all_splits >= total_images_all_splits,
        'total_images': total_images_all_splits,
        'total_processed': total_processed_all_splits,
        'message': 'All images have been processed' if total_processed_all_splits >= total_images_all_splits else 'Some images still need processing'
    }
    
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
    # Initialize Flask app with ngrok configuration
    logger.info("=" * 60)
    logger.info(f"Starting Llama Fire Annotator")
    logger.info(f"Local URL: http://localhost:5003")
    logger.info(f"Ngrok URL: Configure with: ngrok http --domain=llamafire.ngrok.dev 5003")
    logger.info("=" * 60)
    
    app.run(
        host='127.0.0.1',  # Only accept local connections (ngrok will proxy)
        port=5003,  # Use port 5003 instead of 80
        debug=False  # Disable debug mode in production
    )