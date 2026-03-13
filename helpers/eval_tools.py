import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def save_classification_report(report_dict, model_name, out_dir='outputs'):
    """Save classification report dictionary to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(report_dict).transpose()
    csv_path = os.path.join(out_dir, f'{model_name}_classification_report.csv')
    df.to_csv(csv_path, index=True)
    print(f'[INFO] Saved report → {csv_path}')

def save_confusion_matrix(y_true, y_pred, class_names, model_name, out_dir='outputs'):
    """Plot and save confusion matrix heatmap."""
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    file_path = os.path.join(out_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(file_path)
    plt.close()
    print(f'[INFO] Saved confusion matrix → {file_path}')