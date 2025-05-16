# hog_svm.py
# Baseline HOG + SVM implementation for defect detection.

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image
import joblib

from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Directory to save results
RESULTS_DIR = os.path.join('results', 'traditional_ml')
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_hog_features(image, size=(128, 128), pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    """
    Extracts HOG features from a single image.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            image_pil = Image.fromarray(image.astype(np.uint8), 'RGB')
        elif image.ndim == 2:
            image_pil = Image.fromarray(image.astype(np.uint8), 'L')
        else:
            if image.ndim == 3:
                image_pil = Image.fromarray(image[:,:,0].astype(np.uint8), 'L')
            else:
                raise ValueError("Unsupported NumPy array format for HOG feature extraction.")
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise TypeError("Input must be a PIL Image or a NumPy array.")

    image_gray_pil = image_pil.convert('L').resize(size)
    image_gray_np = np.array(image_gray_pil)

    features = hog(
        image_gray_np,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, 
        visualize=False
    )
    return features

def train_evaluate_model(
    X_train, y_train, X_test, y_test, 
    model_type='svm', 
    class_names=None,
    use_grid_search=True,
    use_cross_val=True,
    save_model=True
):
    """
    Trains and evaluates a traditional ML model with optional grid search and cross-validation.
    """
    print(f"\n--- Training and Evaluating {model_type.upper()} ---")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.01, 0.1],
        'classifier__kernel': ['rbf', 'linear']
    }
    
    if use_grid_search:
        print(f"Performing grid search for {model_type.upper()}...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        model = pipeline
    
    if use_cross_val:
        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print("Evaluating on test set...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / len(X_test)
    y_proba = model.predict_proba(X_test)
    
    if class_names:
        predictions_dir = os.path.join(RESULTS_DIR, f'{model_type}_predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        predictions_df = pd.DataFrame({
            'sample_idx': range(len(y_test)),
            'true_label': [class_names[y] for y in y_test],
            'pred_label': [class_names[y] for y in y_pred],
            'confidence': [y_proba[i, y_pred[i]] for i in range(len(y_pred))]
        })
        
        predictions_df.to_csv(os.path.join(RESULTS_DIR, f'{model_type}_predictions.csv'), index=False)
        
        misclassified_dir = os.path.join(RESULTS_DIR, f'{model_type}_misclassified')
        os.makedirs(misclassified_dir, exist_ok=True)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    if class_names:
        report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        report_str = classification_report(y_test, y_pred, target_names=class_names)
    else:
        report = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)
    
    print(f"\n{model_type.upper()} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average inference time per image: {avg_inference_time*1000:.2f} ms")
    print("\nClassification Report:")
    print(report_str)
    
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'model': model,
        'model_type': model_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_test': y_test,
        'y_proba': y_proba,
        'train_time': train_time,
        'inference_time': inference_time,
        'avg_inference_time': avg_inference_time,
    }
    
    if use_grid_search:
        results['best_params'] = grid_search.best_params_
    
    if use_cross_val:
        results['cv_scores'] = cv_scores
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
    
    return results

def visualize_results(results, class_names, feature_type, save_dir):
    """
    Visualizes and saves the results of a traditional ML model.
    """
    model_type = results['model_type']
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    cm = results['confusion_matrix']
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (%) - {feature_type.upper()} + {model_type.upper()}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature_type}_{model_type}_confusion_matrix.png'))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for cls in class_names:
        if cls in results['report']:
            classes.append(cls)
            precision.append(results['report'][cls]['precision'])
            recall.append(results['report'][cls]['recall'])
            f1.append(results['report'][cls]['f1-score'])
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f'Per-class Performance - {feature_type.upper()} + {model_type.upper()}')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{feature_type}_{model_type}_per_class_metrics.png'))
    plt.close()
