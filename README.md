# SteelNet: CNN-Based Defect Detection in Manufacturing Images

This project presents **SteelNet**, a Convolutional Neural Network model built using transfer learning with ResNet18, designed to automate the classification of six steel surface defects from the NEU-DET dataset. The model is evaluated against a classical machine learning approach using Histogram of Oriented Gradients (HOG) combined with a Support Vector Machine (SVM) classifier.

---

### 📊 Key Results

| Model         | Accuracy | F1 Score |
|---------------|----------|----------|
| HOG + SVM     | 92.44%   | 0.9231   |
| SteelNet CNN  | 89.6%    | 0.91     |

---

### 🛠 Features

- **Preprocessing**: Image resizing (224x224), normalization, and augmentation (flip, rotation)
- **Architecture**: ResNet18 with a custom classification head for 6 defect classes
- **Training**: Adam optimizer, Cross-Entropy loss, LR scheduling via ReduceLROnPlateau
- **Classical Baseline**: HOG feature extraction + SVM with RBF kernel
- **Evaluation**: Accuracy, macro-averaged F1-score, per-class metrics

---

### 📂 Dataset

**NEU-DET**: 1,800 grayscale images (6 classes × 300 images), available at  
🔗 [Kaggle NEU Surface Defect Dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)

---

### 📌 Highlights from Report

- SteelNet outperformed HOG + SVM on defect types like *patches* and *pitted surface*
- HOG + SVM showed superior results on *crazing*, *inclusion*, and *rolled-in scale*
- Clear signs of early overfitting in CNN after ~8 epochs
- Future work: Grad-CAM explainability, ensemble models, real-time deployment

---

### 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### SteelNet CNN
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the main entrypoint:
```bash
python main.py
```
3. Follow the prompts to run CNN experiments, traditional ML experiments, or analyze results.

### Traditional ML (HOG+SVM)
The traditional ML pipeline is integrated and can be run via the main entrypoint as well.

## Project Structure
```
steel-defect-detection/
├── main.py                      # Main entrypoint for running experiments
├── models/                      # SteelNet CNN and traditional ML model scripts
├── utils/                       # Preprocessing and data loading utilities
├── results/                     # Final result outputs (confusion matrices, metrics)
├── notebooks/                   # Exploratory analysis notebooks (optional)
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
└── NEU-DET/                    # Dataset directory (not included)
```

## Author
Evan Musick  
Missouri State University

- 🔗 [LinkedIn](https://www.linkedin.com/in/evan-musick-49ba15187)
- 📝 [Medium](https://medium.com/@evanmusick.dev)
- 💻 [GitHub](https://github.com/musickevan1)
