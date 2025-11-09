import matplotlib.pyplot as plt
import re
import os
import glob

def find_latest_log_file(base_path):
    """Find the most recent log file in a directory"""
    # Look for .log files in the directory
    log_files = glob.glob(os.path.join(base_path, "**", "*.log"), recursive=True)
    if not log_files:
        return None
    
    # Return the most recent file (by modification time)
    return max(log_files, key=os.path.getmtime)

def parse_log(filepath):
    epochs = []
    train_acc, val_acc = [], []
    train_prec, val_prec = [], []
    train_rec, val_rec = [], []
    train_f1, val_f1 = [], []

    # Updated regex patterns based on the actual log format
    # Training pattern: Epoch: [X][Y/Z] ... Top1 XX.XXXXXX Top5 XX.XXXXXX
    train_pattern = re.compile(r"Epoch: \[\d+\]\[\s*\d+/\s*\d+\].*?Top1\s+([\d.]+).*?Top5\s+([\d.]+)")
    
    # Validation pattern: ==> Top1: XX.XXX Top5: XX.XXX ... ==> Precision: X.XXX Recall: X.XXX F1: X.XXX
    val_top1_pattern = re.compile(r"==> Top1:\s+([\d.]+).*?Top5:\s+([\d.]+)")
    val_metrics_pattern = re.compile(r"==> Precision:\s+([\d.]+).*?Recall:\s+([\d.]+).*?F1:\s+([\d.]+)")

    with open(filepath, 'r') as f:
        content = f.read()
        
        # Find all training epochs
        train_matches = train_pattern.findall(content)
        for match in train_matches:
            train_acc.append(float(match[0]))
            # For training, we don't have precision/recall/f1 in the same line
            # We'll use the validation values or set to 0
            train_prec.append(0.0)
            train_rec.append(0.0)
            train_f1.append(0.0)
        
        # Find validation metrics
        val_top1_matches = val_top1_pattern.findall(content)
        val_metrics_matches = val_metrics_pattern.findall(content)
        
        for i, match in enumerate(val_top1_matches):
            val_acc.append(float(match[0]))
        
        for i, match in enumerate(val_metrics_matches):
            if i < len(val_acc):  # Ensure we have corresponding accuracy data
                val_prec.append(float(match[0]))
                val_rec.append(float(match[1]))
                val_f1.append(float(match[2]))
        
        # Create epochs list
        epochs = list(range(1, len(train_acc) + 1))

    return {
        'epochs': epochs, 'train_acc': train_acc, 'val_acc': val_acc,
        'train_prec': train_prec, 'val_prec': val_prec, 'train_rec': train_rec,
        'val_rec': val_rec, 'train_f1': train_f1, 'val_f1': val_f1
    }

def plot_all_metrics_for_model(data, model_name, max_epochs=500):
    """Create a single plot showing all metrics for one model, limited to max_epochs"""
    plt.figure(figsize=(12, 8))
    
    # Define distinct colors and styles for different metrics
    metric_styles = {
        'acc': {'color': '#2E86AB', 'linestyle': '-', 'linewidth': 3, 'alpha': 0.8, 'label': 'Accuracy'},
        'prec': {'color': '#A23B72', 'linestyle': '--', 'linewidth': 3, 'alpha': 0.8, 'label': 'Precision'},
        'rec': {'color': '#F18F01', 'linestyle': '-.', 'linewidth': 3, 'alpha': 0.8, 'label': 'Recall'},
        'f1': {'color': '#C73E1D', 'linestyle': ':', 'linewidth': 4, 'alpha': 0.9, 'label': 'F1 Score'}
    }
    
    # Limit data to max_epochs
    limited_epochs = data['epochs'][:max_epochs]
    
    # Plot each metric
    for metric in ['acc', 'prec', 'rec', 'f1']:
        style = metric_styles[metric]
        
        # Use validation data if available, otherwise training data
        y_data = data[f'val_{metric}'] if data[f'val_{metric}'] else data[f'train_{metric}']
        
        if y_data:
            # Limit y_data to max_epochs as well
            limited_y_data = y_data[:max_epochs]
            plt.plot(limited_epochs, limited_y_data, 
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=style['linewidth'],
                    alpha=style['alpha'],
                    label=style['label'])
    
    plt.title(f'{model_name}: All Metrics Over Time (First {len(limited_epochs)} Epochs)', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Add some styling
    plt.tight_layout()
    plt.show()

# --- Main ---
# Updated to use actual directory structure
log_directories = {
    '64x64 Model': 'C:/Users/ndalu/OneDrive/Desktop/AI8X/ai8x-training/logs/handwash-64x64',
    '96x96 Model': 'C:/Users/ndalu/OneDrive/Desktop/AI8X/ai8x-training/logs/handwash-96x96',
    '128x128 Model': 'C:/Users/ndalu/OneDrive/Desktop/AI8X/ai8x-training/logs/handwash-128x128'
}

for model_name, log_dir in log_directories.items():
    print(f"--- Processing {model_name} ---")
    
    # Find the latest log file
    log_file = find_latest_log_file(log_dir)
    if not log_file:
        print(f"Warning: No log file found for {model_name}")
        continue
    
    print(f"Using log file: {log_file}")
    
    try:
        parsed_data = parse_log(log_file)
        
        if not parsed_data['epochs']:
            print(f"Warning: No data parsed for {model_name}. Check log file format and regex patterns.")
            continue

        # Ensure validation data list has the same length as training data list
        for metric in ['acc', 'prec', 'rec', 'f1']:
            while len(parsed_data[f'val_{metric}']) < len(parsed_data[f'train_{metric}']):
                parsed_data[f'val_{metric}'].append(parsed_data[f'val_{metric}'][-1] if parsed_data[f'val_{metric}'] else 0)

        # Create the combined plot for this model (limited to 500 epochs)
        print(f"Creating plot for {model_name} with first {min(len(parsed_data['epochs']), 500)} epochs...")
        plot_all_metrics_for_model(parsed_data, model_name, max_epochs=500)
            
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        continue