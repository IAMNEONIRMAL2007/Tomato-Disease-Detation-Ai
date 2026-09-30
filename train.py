# Created By NirmalBorole
#!/usr/bin/env python3
"""
🍅 Tomato Disease Prediction - Local Training Script
=====================================================
Deep Learning CNN / Transfer Learning model for tomato leaf disease classification.

Usage:
    .\venv\Scripts\python.exe train.py

Diseases detected (10 classes):
    Bacterial Spot, Early Blight, Late Blight, Leaf Mold,
    Septoria Leaf Spot, Spider Mites, Target Spot,
    Mosaic Virus, Yellow Leaf Curl Virus, Healthy
"""

# ============================================================
# 1. Imports
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for local runs
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import random
import datetime
from pathlib import Path
from collections import Counter

# TensorFlow & Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization,
    Activation, Flatten, Dropout, Dense
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard
)

# Scikit-learn
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc,
    accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 2. Configuration
# ============================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ---- EDIT THESE SETTINGS AS NEEDED ----
CONFIG = {
    # Data settings - points to YOUR local dataset
    'data_dir': os.path.join(os.path.dirname(__file__), 'Dataset'),
    'img_size': (224, 224),
    'batch_size': 32,

    # Model settings
    # Options: 'cnn' for baseline, 'transfer_learning' for EfficientNet/MobileNet
    'model_type': 'transfer_learning',
    # Options: 'EfficientNetB0', 'MobileNetV2', 'ResNet50V2'
    'base_model_name': 'EfficientNetB0',
    'num_classes': 10,
    'use_class_weights': True,

    # Training settings
    'epochs_stage1': 15,       # Stage 1: frozen base (transfer learning only)
    'epochs_stage2': 15,       # Stage 2: fine-tune (transfer learning only)
    'epochs_cnn': 30,          # Epochs for baseline CNN
    'learning_rate': 0.001,
    'learning_rate_finetune': 0.0001,
    'optimizer': 'adam',

    # Augmentation
    'use_advanced_augmentation': True,

    # Callbacks
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'min_lr': 1e-7,

    # Output paths
    'model_save_dir': os.path.join(os.path.dirname(__file__), 'models'),
    'logs_dir': os.path.join(os.path.dirname(__file__), 'logs'),
    'output_dir': os.path.join(os.path.dirname(__file__), 'outputs'),
}


def main():
    print("=" * 80)
    print("🍅 TOMATO DISEASE PREDICTION - LOCAL TRAINING")
    print("=" * 80)

    # ============================================================
    # 3. Environment Info
    # ============================================================
    print(f"\nTensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU available: {gpus}")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️  No GPU detected — training will use CPU (slower).")

    # Create directories
    for dir_path in [CONFIG['model_save_dir'], CONFIG['logs_dir'], CONFIG['output_dir']]:
        os.makedirs(dir_path, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Model: {CONFIG['model_type']} with {CONFIG['base_model_name']}")
    print(f"  Image size: {CONFIG['img_size']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Dataset: {CONFIG['data_dir']}")

    # ============================================================
    # 4. Load Dataset
    # ============================================================
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    if not os.path.exists(CONFIG['data_dir']):
        print(f"ERROR: Dataset directory not found at: {CONFIG['data_dir']}")
        print("Please ensure your dataset is in the 'Dataset' folder.")
        return

    full_dataset = tf.keras.utils.image_dataset_from_directory(
        CONFIG['data_dir'],
        image_size=CONFIG['img_size'],
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        seed=SEED,
        label_mode='int'
    )

    class_names = full_dataset.class_names
    CONFIG['num_classes'] = len(class_names)
    print(f"\nDetected classes ({len(class_names)}):")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    # Split dataset: 70% train, 20% val, 10% test
    total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
    train_size = int(0.7 * total_batches)
    val_size = int(0.2 * total_batches)
    test_size = total_batches - train_size - val_size

    train_dataset = full_dataset.take(train_size)
    remaining = full_dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)

    print(f"\nDataset split:")
    print(f"  Training batches:   {train_size} (~{train_size * CONFIG['batch_size']} images)")
    print(f"  Validation batches: {val_size} (~{val_size * CONFIG['batch_size']} images)")
    print(f"  Test batches:       {test_size} (~{test_size * CONFIG['batch_size']} images)")

    # ============================================================
    # 5. Class Distribution
    # ============================================================
    print("\nAnalyzing class distribution...")
    class_counts = {
        name: len([f for f in os.listdir(os.path.join(CONFIG['data_dir'], name))
                   if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for name in class_names
    }

    class_dist_df = pd.DataFrame([
        {'Disease': name, 'Count': count,
         'Percentage': count / sum(class_counts.values()) * 100}
        for name, count in class_counts.items()
    ]).sort_values('Count', ascending=False)

    print("\nClass Distribution:")
    print(class_dist_df.to_string(index=False))

    # Plot class distribution
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(class_dist_df['Disease'], class_dist_df['Count'], color='steelblue')
    axes[0].set_xlabel('Disease Class')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Class Distribution (Training Set)', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].pie(class_dist_df['Count'], labels=class_dist_df['Disease'],
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Distribution (%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'class_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  → Saved class_distribution.png")

    # Class weights
    class_weights = None
    if CONFIG['use_class_weights']:
        total_samples = sum(class_counts.values())
        class_weights = {
            i: total_samples / (len(class_counts) * count)
            for i, (name, count) in enumerate(class_counts.items())
        }

    # ============================================================
    # 6. Data Preprocessing & Augmentation
    # ============================================================
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING & AUGMENTATION")
    print("=" * 80)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)

    if CONFIG['use_advanced_augmentation']:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.3),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2),
        ], name='data_augmentation')
    else:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
        ], name='data_augmentation')

    def preprocess_dataset(dataset, augment=False):
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    train_dataset = preprocess_dataset(train_dataset, augment=True)
    val_dataset = preprocess_dataset(val_dataset, augment=False)
    test_dataset = preprocess_dataset(test_dataset, augment=False)

    # Performance optimization
    train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print("Data preprocessing and augmentation configured!")

    # ============================================================
    # 7. Build Model
    # ============================================================
    print("\n" + "=" * 80)
    print("BUILDING MODEL")
    print("=" * 80)

    input_shape = (*CONFIG['img_size'], 3)
    base_model = None

    if CONFIG['model_type'] == 'transfer_learning':
        print(f"Building Transfer Learning model: {CONFIG['base_model_name']}")

        # Select base model
        if CONFIG['base_model_name'] == 'EfficientNetB0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False,
                                        input_shape=input_shape)
        elif CONFIG['base_model_name'] == 'MobileNetV2':
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                     input_shape=input_shape)
        elif CONFIG['base_model_name'] == 'ResNet50V2':
            base_model = ResNet50V2(weights='imagenet', include_top=False,
                                    input_shape=input_shape)
        else:
            raise ValueError(f"Unknown base model: {CONFIG['base_model_name']}")

        base_model.trainable = False

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(CONFIG['num_classes'], activation='softmax')
        ], name=f"transfer_{CONFIG['base_model_name']}")

    else:  # baseline CNN
        print("Building baseline CNN model")
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            BatchNormalization(), Activation('relu'), MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), padding='same'),
            BatchNormalization(), Activation('relu'), MaxPooling2D((2, 2)),

            Conv2D(128, (3, 3), padding='same'),
            BatchNormalization(), Activation('relu'), MaxPooling2D((2, 2)),

            Conv2D(256, (3, 3), padding='same'),
            BatchNormalization(), Activation('relu'), MaxPooling2D((2, 2)),

            Flatten(),
            Dense(512, activation='relu'), Dropout(0.5),
            Dense(256, activation='relu'), Dropout(0.3),
            Dense(CONFIG['num_classes'], activation='softmax')
        ], name='baseline_cnn')

    # Compile
    if CONFIG['optimizer'] == 'adam':
        optimizer = Adam(learning_rate=CONFIG['learning_rate'])
    else:
        optimizer = SGD(learning_rate=CONFIG['learning_rate'], momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    # ============================================================
    # 8. Callbacks
    # ============================================================
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['early_stopping_patience'],
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=CONFIG['reduce_lr_patience'],
        min_lr=CONFIG['min_lr'], verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(CONFIG['model_save_dir'],
                              'tomato_disease_{epoch:02d}_{val_accuracy:.4f}.h5'),
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    log_dir = os.path.join(CONFIG['logs_dir'],
                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=0)
        callbacks = [early_stopping, reduce_lr, checkpoint, tensorboard_cb]
    except Exception:
        callbacks = [early_stopping, reduce_lr, checkpoint]

    print(f"\nTensorBoard logs: {log_dir}")
    print("  → To view: tensorboard --logdir=./logs")

    # ============================================================
    # 9. Train
    # ============================================================
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    if CONFIG['model_type'] == 'transfer_learning':
        # --- Stage 1: Train new layers only (base frozen) ---
        print(f"\n{'=' * 60}")
        print("STAGE 1: Training new layers (base model frozen)")
        print(f"{'=' * 60}")

        history_stage1 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CONFIG['epochs_stage1'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        best_s1 = max(history_stage1.history['val_accuracy'])
        print(f"\n✅ Stage 1 Complete! Best val_accuracy: {best_s1:.4f}")

        # --- Stage 2: Fine-tune (unfreeze top half of base) ---
        print(f"\n{'=' * 60}")
        print("STAGE 2: Fine-tuning (base model partially unfrozen)")
        print(f"{'=' * 60}")

        base_model.trainable = True
        fine_tune_at = len(base_model.layers) // 2
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        trainable_layers = sum(1 for l in base_model.layers if l.trainable)
        print(f"Unfrozen layers: {trainable_layers} / {len(base_model.layers)}")

        model.compile(
            optimizer=Adam(learning_rate=CONFIG['learning_rate_finetune']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history_stage2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CONFIG['epochs_stage2'],
            callbacks=callbacks,
            class_weight=class_weights,
            initial_epoch=len(history_stage1.history['loss']),
            verbose=1
        )

        best_s2 = max(history_stage2.history['val_accuracy'])
        print(f"\n✅ Stage 2 Complete! Best val_accuracy: {best_s2:.4f}")

        # Combine histories
        history_dict = {
            'loss': history_stage1.history['loss'] + history_stage2.history['loss'],
            'accuracy': history_stage1.history['accuracy'] + history_stage2.history['accuracy'],
            'val_loss': history_stage1.history['val_loss'] + history_stage2.history['val_loss'],
            'val_accuracy': history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy'],
        }
        stage1_epochs = len(history_stage1.history['loss'])

    else:
        # Baseline CNN — single-stage training
        history_result = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=CONFIG['epochs_cnn'],
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        history_dict = history_result.history
        stage1_epochs = None

    # ============================================================
    # 10. Training Visualization
    # ============================================================
    print("\nGenerating training plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(history_dict['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history_dict['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_dict['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history_dict['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    acc_improvement = np.diff([0] + history_dict['val_accuracy'])
    axes[1, 0].bar(range(len(acc_improvement)), acc_improvement,
                   color=['green' if x > 0 else 'red' for x in acc_improvement])
    axes[1, 0].set_title('Validation Accuracy Change per Epoch', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy Change')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Train vs Val gap (overfitting indicator)
    gap = [t - v for t, v in zip(history_dict['accuracy'], history_dict['val_accuracy'])]
    axes[1, 1].plot(gap, linewidth=2, color='orange')
    axes[1, 1].fill_between(range(len(gap)), gap, alpha=0.3, color='orange')
    axes[1, 1].set_title('Train-Val Accuracy Gap (Overfit Indicator)', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3)

    if stage1_epochs is not None:
        for ax in axes.flat[:2]:
            ax.axvline(x=stage1_epochs - 1, color='red', linestyle='--',
                       label='Fine-tuning starts', linewidth=2)
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_history.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  → Saved training_history.png")

    print(f"\nTotal epochs: {len(history_dict['loss'])}")
    print(f"Final train accuracy: {history_dict['accuracy'][-1]:.4f}")
    print(f"Final val accuracy:   {history_dict['val_accuracy'][-1]:.4f}")
    print(f"Best val accuracy:    {max(history_dict['val_accuracy']):.4f}")

    # ============================================================
    # 11. Evaluation
    # ============================================================
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)

    y_true, y_pred, y_pred_proba = [], [], []
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    test_accuracy = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n{'=' * 60}")
    print(f"  Test Accuracy:        {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"  Test F1 (weighted):   {test_f1:.4f}")
    print(f"{'=' * 60}")

    # Classification report
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=class_names, digits=4))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    metrics_df = pd.DataFrame({
        'Disease': class_names,
        'Precision': precision, 'Recall': recall,
        'F1-Score': f1, 'Support': support
    }).sort_values('F1-Score', ascending=False)
    metrics_df.to_csv(os.path.join(CONFIG['output_dir'], 'classification_metrics.csv'),
                      index=False)
    print("  → Saved classification_metrics.csv")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 10})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  → Saved confusion_matrix.png")

    # ROC curves
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    fig, axes_roc = plt.subplots(2, 5, figsize=(25, 10))
    axes_roc = axes_roc.ravel()
    for i, (ax, cn) in enumerate(zip(axes_roc, class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(cn, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    plt.suptitle('ROC Curves - Per Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'roc_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  → Saved roc_curves.png")

    # Macro AUC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    micro_auc = auc(fpr_micro, tpr_micro)
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_true_bin[:, i], y_pred_proba[:, i])[0]
        for i in range(len(class_names))
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
    mean_tpr /= len(class_names)
    macro_auc = auc(all_fpr, mean_tpr)

    print(f"\nMicro-average AUC: {micro_auc:.4f}")
    print(f"Macro-average AUC: {macro_auc:.4f}")

    # Error analysis
    misclassified_idx = np.where(y_true != y_pred)[0]
    print(f"\nMisclassified: {len(misclassified_idx)} / {len(y_true)} "
          f"({len(misclassified_idx) / len(y_true) * 100:.2f}%)")

    if len(misclassified_idx) > 0:
        confusion_pairs = [(class_names[y_true[i]], class_names[y_pred[i]])
                           for i in misclassified_idx]
        most_confused = Counter(confusion_pairs).most_common(10)
        print("\nTop confused class pairs:")
        for (tc, pc), count in most_confused:
            print(f"  {tc} → {pc}: {count} times")

    # ============================================================
    # 12. Save Model
    # ============================================================
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_path = os.path.join(CONFIG['model_save_dir'], 'tomato_disease_final.h5')
    model.save(model_path)
    print(f"  ✓ Keras model: {model_path}")

    saved_model_path = os.path.join(CONFIG['model_save_dir'], 'tomato_disease_saved_model')
    model.save(saved_model_path, save_format='tf')
    print(f"  ✓ SavedModel:  {saved_model_path}")

    # TFLite
    print("  Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(CONFIG['model_save_dir'], 'tomato_disease.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"  ✓ TFLite:      {tflite_path} "
          f"({os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB)")

    # Class names
    class_names_path = os.path.join(CONFIG['model_save_dir'], 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump({'class_names': class_names}, f, indent=2)
    print(f"  ✓ Class names: {class_names_path}")

    # Metadata
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    metadata = {
        'model_info': {
            'type': CONFIG['model_type'],
            'base_model': CONFIG['base_model_name'] if CONFIG['model_type'] == 'transfer_learning' else 'CNN',
            'architecture': model.name,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
        },
        'dataset_info': {
            'num_classes': CONFIG['num_classes'],
            'class_names': class_names,
            'img_size': list(CONFIG['img_size']),
        },
        'performance': {
            'best_val_accuracy': float(max(history_dict['val_accuracy'])),
            'test_accuracy': float(test_accuracy),
            'test_f1_score': float(test_f1),
            'micro_auc': float(micro_auc),
            'macro_auc': float(macro_auc),
        },
        'training_date': datetime.datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
    }
    metadata_path = os.path.join(CONFIG['model_save_dir'], 'tomato_disease_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata:    {metadata_path}")

    # Save training history
    history_df = pd.DataFrame(history_dict)
    history_df.to_csv(os.path.join(CONFIG['output_dir'], 'training_history.csv'),
                      index_label='epoch')

    # ============================================================
    # 13. Final Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"  Model:             {CONFIG['model_type']} ({CONFIG['base_model_name']})")
    print(f"  Parameters:        {total_params:,}")
    print(f"  Best Val Accuracy: {max(history_dict['val_accuracy']) * 100:.2f}%")
    print(f"  Test Accuracy:     {test_accuracy * 100:.2f}%")
    print(f"  Test F1-Score:     {test_f1:.4f}")
    print(f"  Macro AUC:         {macro_auc:.4f}")
    print(f"\nOutputs saved to: {CONFIG['output_dir']}")
    print(f"Models saved to:  {CONFIG['model_save_dir']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
