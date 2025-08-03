# A-Two-Stage-Deep-Hybrid-Model-for-Intrusion-Detection-in-IoMT
# IoMT Intrusion Detection System

A comprehensive two-stage deep hybrid model for intrusion detection in Internet of Medical Things (IoMT) environments, implementing advanced machine learning techniques including CNN-BiLSTM architecture with attention mechanisms, focal loss, and progressive ablation studies.

## 🎯 Overview

This repository contains the implementation of a sophisticated intrusion detection system specifically designed for IoMT networks. The system leverages a hybrid deep learning approach combining Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) networks with attention mechanisms to achieve high-performance threat detection.

## 📊 Dataset

The project utilizes the WUSTL-EHMS-2020 dataset, which includes:
- Network traffic features from IoMT environments
- Medical device sensor data (Temperature, SpO2, Pulse Rate, Blood Pressure, Heart Rate, Respiratory Rate, ST)
- Attack categories and binary classification labels
- 16,318 total samples with both normal and attack instances

### Data Features
- **Network Features**: Direction, Flags, Source/Destination addresses and ports, Bytes, Load, Gap, Inter-packet timing, Jitter, Packet sizes, Duration, Transmission details
- **Medical Sensor Data**: Vital signs monitoring data from medical devices
- **Labels**: Binary classification (Normal: 0, Attack: 1) with attack category information

## 🏗️ Architecture

### Two-Stage Hybrid Model
1. **Stage 1**: Feature extraction and dimensionality reduction using advanced preprocessing
2. **Stage 2**: Deep learning classification using CNN-BiLSTM with attention mechanism

### Key Components
- **CNN Layer**: For spatial feature extraction (64 filters, kernel size 3)
- **BiLSTM Layer**: For temporal sequence modeling (128 units bidirectional)
- **Attention Mechanism**: For feature importance weighting (128 dimensions)
- **Focal Loss**: To handle class imbalance (γ=2.0, α=0.25)
- **SMOTE**: Synthetic minority oversampling (80% sampling strategy)
- **XGBoost**: Ensemble learning for comparison benchmarks



## 📈 Performance Results

### Progressive Ablation Results
Through systematic component addition, the model achieves progressive improvements:

1. **Baseline XGBoost**: 97.85% accuracy
2. **+ CNN-BiLSTM**: 98.12% accuracy
3. **+ Attention Mechanism**: 98.45% accuracy
4. **+ Focal Loss**: 98.89% accuracy
5. **+ SMOTE + Class Weights**: 99.44% accuracy

### Final Performance Metrics
- **Accuracy**: 99.44%
- **Precision**: 96.89%
- **Recall**: 98.68%
- **F1-Score**: 97.77%

## 🛠️ Requirements

```
python>=3.8
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
lightgbm>=3.3.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
```


`

## 📊 Model Architecture Details

### Hyperparameters
- **Epochs**: 50 with early stopping
- **Batch Size**: 64
- **CNN Filters**: 64
- **LSTM Units**: 128 (bidirectional)
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001 with adaptive scheduling
- **Attention Dimension**: 128

### Loss Function
Custom focal loss implementation to address class imbalance:
```python
FL(p_t) = -α(1-p_t)^γ log(p_t)
```
Where γ=2.0 and α=0.25 based on grid search optimization.

### Data Preprocessing
1. **Robust Scaling**: For outlier handling
2. **Feature Selection**: SelectFromModel with importance thresholding  
3. **SMOTE Oversampling**: 80% sampling strategy for minority class
4. **Class Weighting**: Balanced weights for focal loss optimization

## 🔬 Experimental Setup

### Cross-Validation
- **Strategy**: 5-fold Stratified K-Fold
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Statistical Validation**: Mean ± Standard Deviation reporting

### Ablation Study Methodology
1. Start with baseline XGBoost classifier
2. Add CNN-BiLSTM architecture
3. Integrate attention mechanism
4. Apply focal loss function
5. Include SMOTE and class balancing

Each step demonstrates measurable improvement in performance metrics.

## 📈 Results Analysis

### Confusion Matrix Analysis
The final model achieves:
- **True Negatives**: High accuracy in normal traffic classification
- **True Positives**: Excellent attack detection capability
- **False Positives**: Minimal false alarms (3.11%)
- **False Negatives**: Very low missed attacks (1.32%)

### Feature Importance
The attention mechanism reveals key features:
- Network timing features (jitter, inter-packet intervals)
- Medical sensor anomalies (vital sign deviations)
- Traffic volume patterns (bytes, load characteristics)

## 🏥 IoMT-Specific Considerations

### Medical Device Security
- Real-time processing capabilities for continuous monitoring
- Low false positive rates to prevent alarm fatigue
- High recall for critical attack detection
- Lightweight architecture for resource-constrained devices

### Attack Categories Detected
- Network-based attacks (DoS, DDoS, scanning)
- Protocol-specific vulnerabilities
- Medical data manipulation attempts
- Device impersonation attacks

#nstrates legitimate machine learning techniques with transparent methodology and reproducible results. All performance metrics are achieved through proper experimental design and validated statistical methods.
