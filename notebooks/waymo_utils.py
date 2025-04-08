# Waymo Dataset Video Segmentation

This notebook implements a method to identify video transitions in a large collection of images by calculating cosine similarity between consecutive frames.

## Setup and Imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# Try to use tqdm.notebook, but fall back to regular tqdm if not available
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
import torch
from torchvision import transforms, models
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from pathlib import Path
import cv2

# Set matplotlib parameters for better visualization
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('ggplot')

# Define paths
DATA_DIR = "/path/to/waymo/images"  # Update this with your actual path

## Stage 1: Calculating Pairwise Cosine Similarity

We'll implement two methods for feature extraction:
1. A lightweight CNN-based approach using MobileNetV2
2. A simple image resize and flatten approach

You can choose which method to use based on your needs.

### Method 1: CNN-based Feature Extraction

def extract_features_cnn(img_path, model, transform):
    """Extract features from an image using a pre-trained CNN model."""
    try:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_t = batch_t.to(device)
        model = model.to(device)
        
        # Extract features (without computing gradients)
        with torch.no_grad():
            features = model(batch_t)
            
        # Convert to numpy and flatten
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Load a pre-trained model and create a transform
def setup_model():
    # Use MobileNetV2 (lightweight and fast)
    model = models.mobilenet_v2(weights='DEFAULT')
    # Remove the classification layer
    model.classifier = torch.nn.Identity()
    model.eval()
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return model, transform

### Method 2: Simple Resize and Flatten

def extract_features_simple(img_path, size=(128, 128)):
    """Extract features by resizing and flattening the image."""
    try:
        # Use OpenCV for faster processing
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, size)
        # Flatten to 1D array
        features = img.flatten() / 255.0  # Normalize to [0, 1]
        return features
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

### Calculate Similarities for All Image Pairs

def calculate_similarities(img_files, feature_extraction_fn, *args):
    """
    Calculate cosine similarity between consecutive image pairs.
    
    Args:
        img_files: List of image file paths
        feature_extraction_fn: Function to extract features from images
        *args: Additional arguments to pass to feature_extraction_fn
        
    Returns:
        DataFrame with image pairs and their similarity scores
    """
    similarities = []
    features_cache = {}  # Cache features to avoid recomputing
    
    for i in tqdm(range(len(img_files) - 1), desc="Calculating similarities"):
        img1_path = img_files[i]
        img2_path = img_files[i + 1]
        
        # Extract features for first image (or get from cache)
        if img1_path in features_cache:
            features1 = features_cache[img1_path]
        else:
            features1 = feature_extraction_fn(img1_path, *args)
            features_cache[img1_path] = features1
            
        # Extract features for second image
        if img2_path in features_cache:
            features2 = features_cache[img2_path]
        else:
            features2 = feature_extraction_fn(img2_path, *args)
            features_cache[img2_path] = features2
            
        # Skip if either feature extraction failed
        if features1 is None or features2 is None:
            similarities.append({
                'img1': os.path.basename(img1_path),
                'img2': os.path.basename(img2_path),
                'similarity': np.nan
            })
            continue
            
        # Calculate cosine similarity
        similarity = cosine_similarity([features1], [features2])[0][0]
        
        similarities.append({
            'img1': os.path.basename(img1_path),
            'img2': os.path.basename(img2_path),
            'similarity': similarity
        })
        
        # Free up some memory by removing oldest cache entry if cache gets too large
        if len(features_cache) > 100:  # Adjust based on your available memory
            oldest_key = list(features_cache.keys())[0]
            del features_cache[oldest_key]
    
    return pd.DataFrame(similarities)

## Run Stage 1: Feature Extraction and Similarity Calculation

# Get sorted list of image files in the directory
def get_sorted_image_files(data_dir):
    """Get sorted list of image files in the directory."""
    # Find all image files (supporting common formats)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    img_files = []
    for ext in image_extensions:
        img_files.extend(list(Path(data_dir).glob(f'*{ext}')))
        img_files.extend(list(Path(data_dir).glob(f'*{ext.upper()}')))
    
    # Convert to strings and sort
    img_files = [str(f) for f in img_files]
    
    # Natural sort to handle numerical filenames correctly
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]
    
    return sorted(img_files, key=natural_sort_key)

# Choose which method to use:
# 1. CNN-based (more accurate but slower)
# 2. Simple resize (faster but less accurate)

# Uncomment the method you want to use:

# Method 1: CNN-based
# model, transform = setup_model()
# img_files = get_sorted_image_files(DATA_DIR)
# similarity_df = calculate_similarities(img_files, extract_features_cnn, model, transform)

# Method 2: Simple resize (faster)
# img_files = get_sorted_image_files(DATA_DIR)
# similarity_df = calculate_similarities(img_files, extract_features_simple, (128, 128))

# For demo, let's use a small subset to see how it works first
def demo_with_subset(data_dir, n=100):
    """Run with a subset of images for demonstration."""
    img_files = get_sorted_image_files(data_dir)
    
    if len(img_files) > n:
        # Take a subset for demonstration
        img_files = img_files[:n]
        
    print(f"Running with {len(img_files)} images")
    
    # Use the simple method for the demo
    similarity_df = calculate_similarities(img_files, extract_features_simple, (128, 128))
    
    # Save the results
    similarity_df.to_csv('image_similarities_demo.csv', index=False)
    
    return similarity_df

# Uncomment to run the demo
# similarity_df = demo_with_subset(DATA_DIR, n=100)

# To process all 15,000 images, use this:
# img_files = get_sorted_image_files(DATA_DIR)
# similarity_df = calculate_similarities(img_files, extract_features_simple, (128, 128))
# similarity_df.to_csv('image_similarities_full.csv', index=False)

## Stage 2: Analyze Similarities to Determine a Threshold

def analyze_similarities(similarity_df):
    """Analyze the distribution of similarities to help determine a threshold."""
    
    # Plot the distribution of similarity values
    plt.figure(figsize=(12, 6))
    sns.histplot(similarity_df['similarity'], bins=50, kde=True)
    plt.title('Distribution of Cosine Similarity Between Consecutive Frames')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    
    # Plot similarity values over time (frame pairs)
    plt.figure(figsize=(15, 6))
    plt.plot(similarity_df.index, similarity_df['similarity'])
    plt.title('Cosine Similarity Between Consecutive Frames')
    plt.xlabel('Frame Pair Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    
    # Add a horizontal line at the mean - 2*std as a potential threshold
    mean_sim = similarity_df['similarity'].mean()
    std_sim = similarity_df['similarity'].std()
    threshold = mean_sim - 2 * std_sim
    plt.axhline(y=threshold, color='r', linestyle='--', 
                label=f'Potential Threshold: {threshold:.4f} (mean - 2*std)')
    
    # Highlight potential video transitions
    potential_transitions = similarity_df[similarity_df['similarity'] < threshold]
    if not potential_transitions.empty:
        plt.scatter(potential_transitions.index, potential_transitions['similarity'], 
                    color='red', s=50, label=f'Potential Transitions ({len(potential_transitions)})')
    
    plt.legend()
    plt.show()
    
    # Display statistics
    print(f"Similarity Statistics:")
    print(f"  Mean: {mean_sim:.4f}")
    print(f"  Std Dev: {std_sim:.4f}")
    print(f"  Min: {similarity_df['similarity'].min():.4f}")
    print(f"  Max: {similarity_df['similarity'].max():.4f}")
    print(f"  Potential threshold (mean - 2*std): {threshold:.4f}")
    
    # Display first few potential transitions
    if not potential_transitions.empty:
        print(f"\nFirst 10 potential video transitions:")
        display(potential_transitions.head(10))
    else:
        print("\nNo potential transitions found with the suggested threshold.")
    
    return threshold, potential_transitions

# Uncomment to analyze the results
# threshold, potential_transitions = analyze_similarities(similarity_df)

## Stage 3: Identify Video Transitions

def identify_transitions(similarity_df, threshold=None):
    """
    Identify video transitions based on similarity threshold.
    
    Args:
        similarity_df: DataFrame with similarity scores
        threshold: Similarity threshold (if None, automatically calculated as mean - 2*std)
        
    Returns:
        DataFrame with identified transitions
    """
    if threshold is None:
        # Calculate threshold as mean - 2*std if not provided
        mean_sim = similarity_df['similarity'].mean()
        std_sim = similarity_df['similarity'].std()
        threshold = mean_sim - 2 * std_sim
        print(f"Using automatically calculated threshold: {threshold:.4f}")
    
    # Find transitions where similarity drops below threshold
    transitions = similarity_df[similarity_df['similarity'] < threshold].copy()
    
    if transitions.empty:
        print("No transitions found with the current threshold.")
        return transitions
    
    # Add index information for easier reference
    transitions['pair_index'] = transitions.index
    
    # Calculate the frame gap between consecutive transitions
    # This can help identify false positives (transitions too close together)
    if len(transitions) > 1:
        transitions['gap_to_next'] = transitions['pair_index'].diff().shift(-1)
    
    print(f"Found {len(transitions)} potential video transitions")
    
    return transitions

def segment_videos(img_files, transitions):
    """
    Create video segments based on identified transitions.
    
    Args:
        img_files: List of all image files
        transitions: DataFrame with identified transitions
        
    Returns:
        List of video segments (lists of file paths)
    """
    if transitions.empty:
        print("No transitions found, treating all images as one video.")
        return [img_files]
    
    # Get the indices where transitions occur
    transition_indices = transitions['pair_index'].tolist()
    
    # Split into segments
    segments = []
    start_idx = 0
    
    for idx in transition_indices:
        # Add 1 because the transition is between idx and idx+1
        end_idx = idx + 1
        segments.append(img_files[start_idx:end_idx])
        start_idx = end_idx
    
    # Add the last segment
    if start_idx < len(img_files):
        segments.append(img_files[start_idx:])
    
    # Print segment information
    print(f"Created {len(segments)} video segments:")
    for i, segment in enumerate(segments):
        print(f"  Segment {i+1}: {len(segment)} frames")
    
    return segments

def save_segment_info(segments, output_file="video_segments.csv"):
    """Save segment information to a CSV file."""
    
    segment_data = []
    for i, segment in enumerate(segments):
        for j, img_path in enumerate(segment):
            segment_data.append({
                'segment_id': i+1,
                'frame_index': j+1,
                'filename': os.path.basename(img_path),
                'full_path': img_path
            })
    
    segment_df = pd.DataFrame(segment_data)
    segment_df.to_csv(output_file, index=False)
    print(f"Saved segment information to {output_file}")
    
    return segment_df

# Uncomment to run transition detection
# transitions = identify_transitions(similarity_df, threshold=None)
# segments = segment_videos(img_files, transitions)
# segment_df = save_segment_info(segments)

## Visualization of Results

def visualize_transitions(img_files, transitions, n_frames=3):
    """
    Visualize frames before and after potential transitions.
    
    Args:
        img_files: List of all image files
        transitions: DataFrame with identified transitions
        n_frames: Number of frames to show before and after each transition
    """
    # Limit to the first 5 transitions for visualization
    vis_transitions = transitions.head(5) if len(transitions) > 5 else transitions
    
    for i, (idx, row) in enumerate(vis_transitions.iterrows()):
        plt.figure(figsize=(15, 8))
        plt.suptitle(f"Transition {i+1}: {row['img1']} → {row['img2']} (Similarity: {row['similarity']:.4f})")
        
        # Show n frames before and after the transition
        start_idx = max(0, idx - n_frames + 1)
        end_idx = min(len(img_files) - 1, idx + n_frames + 1)
        
        frame_indices = list(range(start_idx, idx + 1)) + list(range(idx + 1, end_idx + 1))
        n_vis_frames = len(frame_indices)
        
        for j, frame_idx in enumerate(frame_indices):
            plt.subplot(2, (n_vis_frames + 1) // 2, j + 1)
            
            if frame_idx == idx:
                plt.title(f"Frame {frame_idx} (Before)")
                border_color = 'blue'
            elif frame_idx == idx + 1:
                plt.title(f"Frame {frame_idx} (After)")
                border_color = 'red'
            else:
                plt.title(f"Frame {frame_idx}")
                border_color = 'black'
            
            img = plt.imread(img_files[frame_idx])
            plt.imshow(img)
            plt.box(on=True)
            plt.gca().spines['bottom'].set_color(border_color)
            plt.gca().spines['top'].set_color(border_color)
            plt.gca().spines['left'].set_color(border_color)
            plt.gca().spines['right'].set_color(border_color)
            plt.gca().spines['bottom'].set_linewidth(5 if frame_idx in [idx, idx+1] else 1)
            plt.gca().spines['top'].set_linewidth(5 if frame_idx in [idx, idx+1] else 1)
            plt.gca().spines['left'].set_linewidth(5 if frame_idx in [idx, idx+1] else 1)
            plt.gca().spines['right'].set_linewidth(5 if frame_idx in [idx, idx+1] else 1)
            plt.xticks([])
            plt.yticks([])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

# Uncomment to visualize transitions
# visualize_transitions(img_files, transitions)

## Complete Pipeline Function

def complete_pipeline(data_dir, method='simple', threshold=None, n_images=None, save_results=True):
    """
    Run the complete pipeline from feature extraction to video segmentation.
    
    Args:
        data_dir: Directory containing image files
        method: 'cnn' or 'simple'
        threshold: Custom similarity threshold (None for automatic)
        n_images: Number of images to process (None for all)
        save_results: Whether to save results to disk
        
    Returns:
        Dictionary with all results
    """
    # Get image files
    img_files = get_sorted_image_files(data_dir)
    if n_images is not None and n_images < len(img_files):
        print(f"Using first {n_images} images out of {len(img_files)}")
        img_files = img_files[:n_images]
    else:
        print(f"Processing all {len(img_files)} images")
    
    # Extract features and calculate similarities
    if method == 'cnn':
        print("Using CNN-based feature extraction")
        model, transform = setup_model()
        similarity_df = calculate_similarities(img_files, extract_features_cnn, model, transform)
    else:
        print("Using simple resize-based feature extraction")
        similarity_df = calculate_similarities(img_files, extract_features_simple, (128, 128))
    
    if save_results:
        similarity_df.to_csv('image_similarities.csv', index=False)
        print("Saved similarities to image_similarities.csv")
    
    # Analyze similarities
    auto_threshold, _ = analyze_similarities(similarity_df)
    
    # Use provided threshold or automatic one
    final_threshold = threshold if threshold is not None else auto_threshold
    print(f"Using threshold: {final_threshold:.4f}")
    
    # Identify transitions
    transitions = identify_transitions(similarity_df, threshold=final_threshold)
    
    if save_results and not transitions.empty:
        transitions.to_csv('video_transitions.csv', index=False)
        print("Saved transitions to video_transitions.csv")
    
    # Create video segments
    segments = segment_videos(img_files, transitions)
    
    if save_results:
        segment_df = save_segment_info(segments)
    
    # Visualize some transitions
    if not transitions.empty:
        visualize_transitions(img_files, transitions)
    
    return {
        'img_files': img_files,
        'similarity_df': similarity_df,
        'threshold': final_threshold,
        'transitions': transitions,
        'segments': segments
    }

# Batch processing function for large datasets
def batch_process_images(data_dir, batch_size=1000, method='simple', save_intermediate=True):
    """
    Process large datasets in batches to avoid memory issues.
    
    Args:
        data_dir: Directory containing image files
        batch_size: Number of images to process in each batch
        method: 'cnn' or 'simple'
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Combined results dictionary
    """
    # Get all image files
    all_img_files = get_sorted_image_files(data_dir)
    total_images = len(all_img_files)
    print(f"Total images found: {total_images}")
    
    # Process in batches
    all_similarities = []
    
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        print(f"\nProcessing batch {batch_start//batch_size + 1}: images {batch_start+1} to {batch_end}")
        
        # Get current batch of images
        img_files_batch = all_img_files[batch_start:batch_end]
        
        # Extract features and calculate similarities
        if method == 'cnn':
            model, transform = setup_model()
            batch_similarities = calculate_similarities(img_files_batch, extract_features_cnn, model, transform)
        else:
            batch_similarities = calculate_similarities(img_files_batch, extract_features_simple, (128, 128))
        
        # Save intermediate results if requested
        if save_intermediate:
            batch_similarities.to_csv(f'similarities_batch_{batch_start//batch_size + 1}.csv', index=False)
        
        # Store results
        all_similarities.append(batch_similarities)
    
    # Combine all similarities
    combined_df = pd.concat(all_similarities, ignore_index=True)
    
    # Save combined results
    combined_df.to_csv('all_image_similarities.csv', index=False)
    print("Saved all similarities to all_image_similarities.csv")
    
    # Analyze and process the combined results
    auto_threshold, _ = analyze_similarities(combined_df)
    transitions = identify_transitions(combined_df, threshold=auto_threshold)
    
    if not transitions.empty:
        transitions.to_csv('video_transitions.csv', index=False)
        print("Saved transitions to video_transitions.csv")
        
        # Create video segments
        segments = segment_videos(all_img_files, transitions)
        segment_df = save_segment_info(segments)
    else:
        segments = [all_img_files]
        print("No transitions found, treating all images as one video")
    
    return {
        'img_files': all_img_files,
        'similarity_df': combined_df,
        'threshold': auto_threshold,
        'transitions': transitions,
        'segments': segments
    }

# Example usage:
# results = complete_pipeline(DATA_DIR, method='simple', n_images=1000)

## Threshold Adjustment

def manual_threshold_adjustment(similarity_df, img_files, thresholds=None):
    """
    Manually test different thresholds and see the resulting transitions.
    This function doesn't require ipywidgets.
    
    Args:
        similarity_df: DataFrame with similarity scores
        img_files: List of image files
        thresholds: List of threshold values to try
    """
    if thresholds is None:
        # Generate some reasonable thresholds based on the data
        mean_sim = similarity_df['similarity'].mean()
        std_sim = similarity_df['similarity'].std()
        base_threshold = mean_sim - 2 * std_sim
        
        # Create 5 thresholds around the base threshold
        thresholds = [
            base_threshold - 2 * std_sim, 
            base_threshold - std_sim,
            base_threshold,
            base_threshold + std_sim,
            base_threshold + 2 * std_sim
        ]
    
    for threshold in thresholds:
        transitions = similarity_df[similarity_df['similarity'] < threshold].copy()
        transitions['pair_index'] = transitions.index
        
        plt.figure(figsize=(15, 6))
        plt.plot(similarity_df.index, similarity_df['similarity'])
        plt.axhline(y=threshold, color='r', linestyle='--', 
                    label=f'Threshold: {threshold:.4f}')
        
        if not transitions.empty:
            plt.scatter(transitions.index, transitions['similarity'], 
                        color='red', s=50, 
                        label=f'Transitions ({len(transitions)})')
        
        plt.title('Cosine Similarity Between Consecutive Frames')
        plt.xlabel('Frame Pair Index')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Found {len(transitions)} transitions with threshold {threshold:.4f}")
        
        # Show a few example transitions
        if not transitions.empty:
            n_examples = min(3, len(transitions))
            example_indices = np.linspace(0, len(transitions)-1, n_examples, dtype=int)
            
            for i in example_indices:
                idx = transitions.iloc[i]['pair_index']
                sim = transitions.iloc[i]['similarity']
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Transition {i+1}/{len(transitions)}: Similarity = {sim:.4f}")
                
                img1 = plt.imread(img_files[idx])
                img2 = plt.imread(img_files[idx+1])
                
                ax1.imshow(img1)
                ax1.set_title(f"Frame {idx}")
                ax1.axis('off')
                
                ax2.imshow(img2)
                ax2.set_title(f"Frame {idx+1}")
                ax2.axis('off')
                
                plt.tight_layout()
                plt.show()
    
    return transitions

def remove_transition(transitions_df, row_idx):
    """
    Remove a specific transition from the DataFrame and update gap_to_next for consistency.
    
    Args:
        transitions_df: DataFrame containing transitions
        row_idx: Index of the transition to remove (integer position in the DataFrame)
        
    Returns:
        Updated DataFrame with the transition removed
    """
    # Make a copy to avoid modifying the original
    df = transitions_df.copy()
    
    if row_idx < 0 or row_idx >= len(df):
        print(f"Error: Index {row_idx} is out of bounds for DataFrame with {len(df)} rows")
        return df
    
    # Get the pair_index of the row to be removed
    removed_pair_idx = df.iloc[row_idx]['pair_index']
    
    # If this isn't the last row and there's a previous row, update gap_to_next for the previous row
    if 'gap_to_next' in df.columns:
        if row_idx > 0 and row_idx < len(df) - 1:
            # Calculate the new gap for the previous row (which should now point to the next row's pair_index)
            prev_idx = row_idx - 1
            next_idx = row_idx + 1
            
            next_pair_idx = df.iloc[next_idx]['pair_index']
            prev_pair_idx = df.iloc[prev_idx]['pair_index']
            
            # Update the gap_to_next for the previous row
            df.at[df.index[prev_idx], 'gap_to_next'] = next_pair_idx - prev_pair_idx
    
    # Remove the row
    df = df.drop(df.index[row_idx])
    
    # Reset index if needed
    df = df.reset_index(drop=True)
    
    print(f"Removed transition at pair index {removed_pair_idx}")
    print(f"Remaining transitions: {len(df)}")
    
    return df

# Example usage:
# transitions = manual_threshold_adjustment(similarity_df, img_files)
# Or with custom thresholds:
# transitions = manual_threshold_adjustment(similarity_df, img_files, thresholds=[0.7, 0.8, 0.9])