#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from google.cloud import storage
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

# Import your model and dataset classes
from .vmaf_data_loader import VMAFDataset
from .model import VMAFNet

def _to_py(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    raise TypeError(f"type {type(o).__name__} not JSON serializable")

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

def load_model(checkpoint_path, device):
    """Load the best trained model"""
    model = VMAFNet(
        width=0.25,
        dropout=0.1,
        activation='leaky_relu',
        norm_groups=16,
        kernel_size=3,
        reduction=16,
        num_conv_layers=4
    ).to(device)
    
    # Download checkpoint if it's from GCS
    if checkpoint_path.startswith("gs://"):
        client = storage.Client()
        uri = checkpoint_path[len("gs://"):]
        bucket_name, _, blob_name = uri.partition("/")
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        local_checkpoint = "/tmp/best_model.pth"
        blob.download_to_filename(local_checkpoint)
        checkpoint_path = local_checkpoint
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model

def run_inference(model, test_loader, device):
    """Run inference on test set"""
    predictions, targets = [], []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if batch is None:
                continue
                
            ref, dist, target = batch
            ref = ref.to(device)
            dist = dist.to(device)
            
            # Skip batch if any tensor has NaN/Inf
            if not (torch.isfinite(ref).all() and torch.isfinite(dist).all() and torch.isfinite(target).all()):
                print("Tensor has NaN/Inf")
                continue
            
            pred = model(ref, dist).view(-1)
            
            # Convert to VMAF scale (0-100)
            predictions.extend((pred * 100).cpu().numpy())
            targets.extend((target * 100).cpu().numpy())
    
    return np.array(predictions), np.array(targets)

def calculate_metrics(predictions, targets):
    """Calculate comprehensive evaluation metrics"""
    # Core metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mse = np.mean((predictions - targets) ** 2)
    
    # Correlation metrics
    plcc = pearsonr(predictions, targets)[0] if len(predictions) > 1 else 0.0
    srcc = spearmanr(predictions, targets)[0] if len(predictions) > 1 else 0.0
    
    # Additional metrics
    mape = np.mean(np.abs((predictions - targets) / np.maximum(targets, 1e-8))) * 100
    r2 = r2_score(targets, predictions)
    
    # Error statistics
    errors = predictions - targets
    error_std = np.std(errors)
    error_median = np.median(np.abs(errors))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'PLCC': plcc,
        'SRCC': srcc,
        'MAPE': mape,
        'R2': r2,
        'Error_Std': error_std,
        'Median_AE': error_median,
        'Min_Error': np.min(errors),
        'Max_Error': np.max(errors)
    }

def error_analysis_by_quality(predictions, targets):
    """Analyze errors by VMAF quality ranges"""
    quality_ranges = [
        (0, 30, 'Poor'),
        (30, 50, 'Fair'), 
        (50, 70, 'Good'),
        (70, 85, 'Very Good'),
        (85, 100, 'Excellent')
    ]
    
    analysis = {}
    
    for min_val, max_val, name in quality_ranges:
        mask = (targets >= min_val) & (targets < max_val)
        if np.any(mask):
            pred_subset = predictions[mask]
            target_subset = targets[mask]
            
            analysis[name] = {
                'count': np.sum(mask),
                'mae': np.mean(np.abs(pred_subset - target_subset)),
                'rmse': np.sqrt(np.mean((pred_subset - target_subset) ** 2)),
                'plcc': pearsonr(pred_subset, target_subset)[0] if len(pred_subset) > 1 else 0.0
            }
    
    return analysis

def create_visualizations(predictions, targets, output_dir):
    """Create evaluation plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    
    # 1. Scatter plot: predicted vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics for title
    mae = np.mean(np.abs(predictions - targets))
    plcc = pearsonr(predictions, targets)[0]
    
    plt.xlabel('True VMAF Score', fontsize=12)
    plt.ylabel('Predicted VMAF Score', fontsize=12)
    plt.title(f'VMAF Prediction Performance\nPLCC: {plcc:.3f}, MAE: {mae:.2f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution
    errors = predictions - targets
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean Error: {np.mean(errors):.2f}')
    plt.xlabel('Prediction Error (Predicted - True)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Error vs True VMAF
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, errors, alpha=0.6, s=20)
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label='No Error')
    plt.xlabel('True VMAF Score', fontsize=12)
    plt.ylabel('Prediction Error', fontsize=12)
    plt.title('Prediction Error vs True VMAF Score', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'error_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Absolute error vs True VMAF
    abs_errors = np.abs(errors)
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, abs_errors, alpha=0.6, s=20)
    plt.xlabel('True VMAF Score', fontsize=12)
    plt.ylabel('Absolute Prediction Error', fontsize=12)
    plt.title('Absolute Prediction Error vs True VMAF Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'abs_error_vs_true.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_results(metrics, quality_analysis):
    """Print comprehensive results"""
    print("\n" + "="*60)
    print("                VMAF MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOVERALL PERFORMANCE METRICS:")
    print(f"  Mean Absolute Error (MAE):     {metrics['MAE']:.3f}")
    print(f"  Root Mean Square Error (RMSE): {metrics['RMSE']:.3f}")
    print(f"  Mean Square Error (MSE):       {metrics['MSE']:.6f}")
    print(f"  Pearson Correlation (PLCC):    {metrics['PLCC']:.4f}")
    print(f"  Spearman Correlation (SRCC):   {metrics['SRCC']:.4f}")
    print(f"  Mean Absolute Percentage Error: {metrics['MAPE']:.2f}%")
    print(f"  R-squared Score:               {metrics['R2']:.4f}")
    
    print(f"\nERROR STATISTICS:")
    print(f"  Standard Deviation of Errors:  {metrics['Error_Std']:.3f}")
    print(f"  Median Absolute Error:         {metrics['Median_AE']:.3f}")
    print(f"  Min Error:                     {metrics['Min_Error']:.3f}")
    print(f"  Max Error:                     {metrics['Max_Error']:.3f}")
    
    print(f"\nPERFORMANCE BY QUALITY RANGE:")
    for quality, stats in quality_analysis.items():
        print(f"  {quality:12} (n={stats['count']:4d}): MAE={stats['mae']:.2f}, "
              f"RMSE={stats['rmse']:.2f}, PLCC={stats['plcc']:.3f}")
    
    print(f"\nCOMPARISON TO ACADEMIC BENCHMARKS:")
    academic_rmse = 4.0
    improvement_needed = (metrics['RMSE'] / academic_rmse - 1) * 100
    print(f"  Academic benchmark RMSE:       {academic_rmse:.1f}")
    print(f"  Your model RMSE:               {metrics['RMSE']:.3f}")
    print(f"  Gap:                           {improvement_needed:+.1f}% higher error")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate VMAF proxy model")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with test data")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory with video frames")
    parser.add_argument("--checkpoint", type=str, default="gs://vmaf_proxy_training_checkpoints/checkpoints/model.pth", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    # Everything is local (prefetched with gsutil rsync to /data/dataset)
    def _gs_to_local(p: str, local_root="/data"):
        if not isinstance(p, str) or not p.startswith("gs://"):
            return p
        # gs://<bucket>/<suffix>  → /data/<suffix>
        bucket_and_suffix = p[5:]
        _, _, suffix = bucket_and_suffix.partition("/")
        return f"{local_root}/{suffix}".rstrip("/")

    # Redirect any gs:// paths to the local mirror
    args.root_dir = _gs_to_local(args.root_dir, "/data")
    args.csv_file = _gs_to_local(args.csv_file, "/data")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = VMAFDataset(csv_file=args.csv_file, root_dir=args.root_dir, crop_size=128, fixed_crop=True, n_frames=3, split="Testing")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),  persistent_workers=(args.num_workers > 0), collate_fn=collate_skip_none)

    print(f"Test dataset size: {len(test_dataset)} samples")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)
    
    # Run inference
    predictions, targets = run_inference(model, test_loader, args.device)
    
    if len(predictions) == 0:
        print("ERROR: No valid predictions generated. Check your test data.")
        return
    
    print(f"Generated {len(predictions)} predictions")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    quality_analysis = error_analysis_by_quality(predictions, targets)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(predictions, targets, args.output_dir)
    
    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    results = {
        'metrics': metrics,
        'quality_analysis': quality_analysis,
        'test_samples': len(predictions),
        'model_config': {
            'width': 0.25,
            'num_conv_layers': 4,
            'crop_size': 128,
            'n_frames': 3
        }
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=_to_py)
    
    # Save raw predictions
    np.save(output_dir / 'predictions.npy', predictions)
    np.save(output_dir / 'targets.npy', targets)
    
    # Print results
    print_results(metrics, quality_analysis)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("Generated files:")
    print("  - evaluation_results.json (detailed metrics)")
    print("  - prediction_scatter.png (scatter plot)")
    print("  - error_distribution.png (error histogram)")
    print("  - error_vs_true.png (error vs true VMAF)")
    print("  - abs_error_vs_true.png (absolute error analysis)")
    print("  - predictions.npy (raw predictions)")
    print("  - targets.npy (ground truth values)")

if __name__ == "__main__":
    main()