# SteelNet: Defect Detection in Steel with CNNs

## Overview
This project implements defect detection in steel surfaces using the NEU-DET dataset. It compares a deep learning approach (SteelNet CNN) with a traditional machine learning baseline (HOG+SVM). The objective is to accurately classify different types of surface defects to improve quality control in steel manufacturing.

## Dataset
The NEU-DET dataset contains images of six types of steel surface defects. The dataset is publicly available [here](https://github.com/erhwenkuo/NEU-DET).

## Models and Results
| Model          | Accuracy   | F1 Score  |
|----------------|------------|-----------|
| HOG+SVM        | 92.44%     | 92.31%    |
| SteelNet (CNN) | [Insert Accuracy] | [Insert F1 Score] |

## Run Instructions

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

## License
This project is licensed under the MIT License.

## Author
Evan Musick  
Missouri State University
