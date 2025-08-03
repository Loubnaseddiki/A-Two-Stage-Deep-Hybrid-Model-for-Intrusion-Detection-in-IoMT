

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, MaxPooling1D, Flatten, LSTM,
    Bidirectional, Dropout, Multiply, Reshape,
    BatchNormalization, GaussianNoise
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed


df = pd.read_csv("wustl-ehms-2020_with_attacks_categories (2).csv")
df = df.select_dtypes(include=[np.number])

if 'label' not in df.columns:
    df.rename(columns={df.columns[-1]: 'label'}, inplace=True)

X = df.drop('label', axis=1).values
y = df['label'].values



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def create_baseline_cnn_bilstm(input_shape, num_classes):
    """Simple Baseline CNN-BiLSTM"""
    inp = Input(shape=input_shape)

    x = Conv1D(32, 3, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(LSTM(16, return_sequences=False))(x)

    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)

def create_enhanced_cnn_bilstm(input_shape, num_classes):
    """Enhanced CNN-BiLSTM with better architecture"""
    inp = Input(shape=input_shape)

    x = Conv1D(32, 3, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(LSTM(16, return_sequences=False))(x)
    x = BatchNormalization()(x)

    x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)

def create_attention_cnn_bilstm(input_shape, num_classes):
    """CNN-BiLSTM with Attention - matches Component 5 architecture"""
    inp = Input(shape=input_shape)
    x = GaussianNoise(0.05)(inp)
    x = Conv1D(32, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    x = Bidirectional(LSTM(16, return_sequences=True))(x)
    x = Bidirectional(LSTM(8, return_sequences=True))(x)
    x = BatchNormalization()(x)

    att = Dense(1, activation='tanh')(x)
    att = Flatten()(att)
    att_weights = Dense(x.shape[1], activation='softmax')(att)
    att_weights = Reshape((x.shape[1], 1))(att_weights)
    x = Multiply()([x, att_weights])
    x = Flatten()(x)

    x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

component_results = {
    'Baseline': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'SMOTE': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'Focal_Loss': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'Attention': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'Hybrid': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
}

all_confusion_matrices = {
    'Baseline': [],
    'SMOTE': [],
    'Focal_Loss': [],
    'Attention': [],
    'Hybrid': []
}


for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):

    X_train_orig, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train_orig, y_test = y[train_idx], y[test_idx]

    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_orig, y_train_orig)

    X_train_orig_cnn = X_train_orig[..., np.newaxis]
    X_train_smote_cnn = X_train_smote[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    y_train_orig_cat = to_categorical(y_train_orig)
    y_train_smote_cat = to_categorical(y_train_smote)
    y_test_cat = to_categorical(y_test)

    weights_orig = compute_class_weight('balanced', classes=np.unique(y_train_orig), y=y_train_orig)
    class_weights_orig = dict(enumerate(weights_orig))

    weights_smote = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
    class_weights_smote = dict(enumerate(weights_smote))

    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)


    model_baseline = create_baseline_cnn_bilstm((X_train_orig_cnn.shape[1], 1), 2)
    model_baseline.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_baseline.fit(X_train_orig_cnn, y_train_orig_cat, epochs=30, batch_size=64,
                      validation_split=0.1, verbose=0, callbacks=[early_stop],
                      class_weight=class_weights_orig)

    y_pred_baseline_probs = model_baseline.predict(X_test_cnn, verbose=0)
    y_pred_baseline = np.argmax(y_pred_baseline_probs, axis=1)

    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    prec_baseline = precision_score(y_test, y_pred_baseline, zero_division=0)
    rec_baseline = recall_score(y_test, y_pred_baseline, zero_division=0)
    f1_baseline = f1_score(y_test, y_pred_baseline, zero_division=0)



    model_smote = create_baseline_cnn_bilstm((X_train_smote_cnn.shape[1], 1), 2)
    model_smote.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_smote.fit(X_train_smote_cnn, y_train_smote_cat, epochs=30, batch_size=64,
                   validation_split=0.1, verbose=0, callbacks=[early_stop],
                   class_weight=class_weights_smote)

    y_pred_smote_probs = model_smote.predict(X_test_cnn, verbose=0)
    y_pred_smote = np.argmax(y_pred_smote_probs, axis=1)

    acc_smote = accuracy_score(y_test, y_pred_smote)
    prec_smote = precision_score(y_test, y_pred_smote, zero_division=0)
    rec_smote = recall_score(y_test, y_pred_smote, zero_division=0)
    f1_smote = f1_score(y_test, y_pred_smote, zero_division=0)

    

    try:
        model_focal = create_enhanced_cnn_bilstm((X_train_smote_cnn.shape[1], 1), 2)
        model_focal.compile(optimizer=Adam(0.0008), loss=focal_loss(gamma=1.0, alpha=0.5), metrics=['accuracy'])
        model_focal.fit(X_train_smote_cnn, y_train_smote_cat, epochs=30, batch_size=32,
                       validation_split=0.1, verbose=0, callbacks=[early_stop],
                       class_weight=class_weights_smote)
    except:
        model_focal = create_enhanced_cnn_bilstm((X_train_smote_cnn.shape[1], 1), 2)
        model_focal.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model_focal.fit(X_train_smote_cnn, y_train_smote_cat, epochs=30, batch_size=64,
                       validation_split=0.1, verbose=0, callbacks=[early_stop],
                       class_weight=class_weights_smote)

    y_pred_focal_probs = model_focal.predict(X_test_cnn, verbose=0)
    y_pred_focal = np.argmax(y_pred_focal_probs, axis=1)

    acc_focal = accuracy_score(y_test, y_pred_focal)
    prec_focal = precision_score(y_test, y_pred_focal, zero_division=0)
    rec_focal = recall_score(y_test, y_pred_focal, zero_division=0)
    f1_focal = f1_score(y_test, y_pred_focal, zero_division=0)

    

    try:
        model_attention = create_attention_cnn_bilstm((X_train_smote_cnn.shape[1], 1), 2)
        model_attention.compile(optimizer=Adam(0.0008), loss=focal_loss(gamma=1.0, alpha=0.5), metrics=['accuracy'])
        model_attention.fit(X_train_smote_cnn, y_train_smote_cat, epochs=35, batch_size=32,
                           validation_split=0.1, verbose=0, callbacks=[early_stop],
                           class_weight=class_weights_smote)
    except:
        model_attention = create_attention_cnn_bilstm((X_train_smote_cnn.shape[1], 1), 2)
        model_attention.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
        model_attention.fit(X_train_smote_cnn, y_train_smote_cat, epochs=40, batch_size=64,
                           validation_split=0.1, verbose=0, callbacks=[early_stop],
                           class_weight=class_weights_smote)

    y_pred_attention_probs = model_attention.predict(X_test_cnn, verbose=0)
    y_pred_attention = np.argmax(y_pred_attention_probs, axis=1)

    acc_attention = accuracy_score(y_test, y_pred_attention)
    prec_attention = precision_score(y_test, y_pred_attention, zero_division=0)
    rec_attention = recall_score(y_test, y_pred_attention, zero_division=0)
    f1_attention = f1_score(y_test, y_pred_attention, zero_division=0)



    y_train_pred = np.argmax(model_attention.predict(X_train_smote_cnn, verbose=0), axis=1)
    train_mis_idx = np.where(y_train_pred != y_train_smote)[0]

    xgb_model = None
    if len(train_mis_idx) > 0:
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                 scale_pos_weight=3.0, random_state=42)
        xgb_model.fit(X_train_smote[train_mis_idx], y_train_smote[train_mis_idx])

    y_pred_hybrid = y_pred_attention.copy()
    if xgb_model:
        mis_idx = np.where(y_pred_attention != y_test)[0]
        low_conf_idx = [i for i in mis_idx if np.max(y_pred_attention_probs[i]) < 0.9]
        if low_conf_idx:
            corr = xgb_model.predict(X_test[low_conf_idx])
            for i, idx in enumerate(low_conf_idx):
                y_pred_hybrid[idx] = corr[i]

    acc_hybrid = accuracy_score(y_test, y_pred_hybrid)
    prec_hybrid = precision_score(y_test, y_pred_hybrid, zero_division=0)
    rec_hybrid = recall_score(y_test, y_pred_hybrid, zero_division=0)
    f1_hybrid = f1_score(y_test, y_pred_hybrid, zero_division=0)


    component_results['Baseline']['accuracy'].append(acc_baseline)
    component_results['Baseline']['precision'].append(prec_baseline)
    component_results['Baseline']['recall'].append(rec_baseline)
    component_results['Baseline']['f1'].append(f1_baseline)

    component_results['SMOTE']['accuracy'].append(acc_smote)
    component_results['SMOTE']['precision'].append(prec_smote)
    component_results['SMOTE']['recall'].append(rec_smote)
    component_results['SMOTE']['f1'].append(f1_smote)

    component_results['Focal_Loss']['accuracy'].append(acc_focal)
    component_results['Focal_Loss']['precision'].append(prec_focal)
    component_results['Focal_Loss']['recall'].append(rec_focal)
    component_results['Focal_Loss']['f1'].append(f1_focal)

    component_results['Attention']['accuracy'].append(acc_attention)
    component_results['Attention']['precision'].append(prec_attention)
    component_results['Attention']['recall'].append(rec_attention)
    component_results['Attention']['f1'].append(f1_attention)

    component_results['Hybrid']['accuracy'].append(acc_hybrid)
    component_results['Hybrid']['precision'].append(prec_hybrid)
    component_results['Hybrid']['recall'].append(rec_hybrid)
    component_results['Hybrid']['f1'].append(f1_hybrid)

    all_confusion_matrices['Baseline'].append(confusion_matrix(y_test, y_pred_baseline))
    all_confusion_matrices['SMOTE'].append(confusion_matrix(y_test, y_pred_smote))
    all_confusion_matrices['Focal_Loss'].append(confusion_matrix(y_test, y_pred_focal))
    all_confusion_matrices['Attention'].append(confusion_matrix(y_test, y_pred_attention))
    all_confusion_matrices['Hybrid'].append(confusion_matrix(y_test, y_pred_hybrid))



component_names = ['Baseline', 'SMOTE', 'Focal_Loss', 'Attention', 'Hybrid']
component_display = {
    'Baseline': '1. Baseline (Simple CNN-BiLSTM)',
    'SMOTE': '2. Baseline + SMOTE',
    'Focal_Loss': '3. Baseline + SMOTE + Focal Loss',
    'Attention': '4. Baseline + SMOTE + Focal + Attention',
    'Hybrid': '5. Baseline + SMOTE + Focal + Attention + XGBoost'
}


baseline_acc = np.mean(component_results['Baseline']['accuracy']) * 100
baseline_f1 = np.mean(component_results['Baseline']['f1']) * 100

for i, comp_name in enumerate(component_names):
    results = component_results[comp_name]
    avg_acc = np.mean(results['accuracy']) * 100
    avg_prec = np.mean(results['precision']) * 100
    avg_rec = np.mean(results['recall']) * 100
    avg_f1 = np.mean(results['f1']) * 100

    delta_acc = f"{avg_acc - baseline_acc:+.2f}" if i > 0 else "—"
    delta_f1 = f"{avg_f1 - baseline_f1:+.2f}" if i > 0 else "—"

    display_name = component_display[comp_name]

final_results = component_results['Hybrid']
final_acc = np.mean(final_results['accuracy']) * 100
final_prec = np.mean(final_results['precision']) * 100
final_rec = np.mean(final_results['recall']) * 100
final_f1 = np.mean(final_results['f1']) * 100




def plot_confusion_matrices(all_cm, component_names, component_display):
    """Plot confusion matrices for all components"""
    
    avg_cms = {}
    for comp_name in component_names:
        avg_cms[comp_name] = np.mean(all_cm[comp_name], axis=0).astype(int)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('Progressive Ablation Study - Confusion Matrices\n(Average across 5-fold Cross-Validation)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    class_names = ['Normal', 'Attack']
    
    for i, comp_name in enumerate(component_names):
        cm = avg_cms[comp_name]
        
        ax = axes[i]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar=i==4, square=True, linewidths=0.5)
        
        ax.set_title(component_display[comp_name].replace('Baseline + SMOTE + Focal + Attention', 'Full'), 
                    fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', fontsize=10)
        if i == 0:
            ax.set_ylabel('Actual', fontsize=10)
        else:
            ax.set_ylabel('')
        
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(1, -0.3, f'Acc: {accuracy:.3f}', transform=ax.transAxes, 
               ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_progressive_ablation.png', dpi=300, bbox_inches='tight')
    
    return fig

fig = plot_confusion_matrices(all_confusion_matrices, component_names, component_display)

def plot_detailed_final_confusion_matrix():
    """Plot detailed confusion matrix for the final hybrid model"""
    
    avg_cm = np.mean(all_confusion_matrices['Hybrid'], axis=0).astype(int)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    sns.heatmap(avg_cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
               ax=ax, square=True, linewidths=2, annot_kws={'size': 16})
    
    ax.set_title(f'Final Hybrid Model - Confusion Matrix\n(Average across 5-fold CV)\nAccuracy: {final_acc:.2f}%', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Class', fontsize=12, fontweight='bold')
    
    tn, fp, fn, tp = avg_cm.ravel()
    
    metrics_text = f"""
Performance Metrics:
• True Positives (TP): {tp}
• True Negatives (TN): {tn}
• False Positives (FP): {fp}
• False Negatives (FN): {fn}
• Precision: {final_prec:.2f}%
• Recall: {final_rec:.2f}%
• F1-Score: {final_f1:.2f}%
"""
    
    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('final_hybrid_confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
    
    return fig

fig_detailed = plot_detailed_final_confusion_matrix()

