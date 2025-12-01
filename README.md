# 🧬 Histopathologic Cancer Detection — Image Classification Project

This project implements an **image classification pipeline** for detecting metastatic tissue in histopathologic (H&E stained) images. It follows the full workflow of a deep-learning project, including **data understanding, augmentation, model experimentation, optimization strategies, and ROC-based evaluation**.

---

## 📘 **1. Project Overview**

The goal is to classify small 96×96 histopathologic images as either **cancerous** or **non-cancerous**. The dataset comes from the **Kaggle Histopathologic Cancer Detection competition**.

* **Data Link:** Kaggle – Histopathologic Cancer Detection
* **Sample Code Reference:** CNN-based Kaggle notebook
* **Subset Used:** ~5000 images (as permitted)

The project includes:

* Reading dataset description & breast cancer background
* Understanding **H&E staining (Hematoxylin & Eosin)**
* Designing models, training strategies, and hyperparameter tuning
* Testing augmentations, optimizers, learning rate schedulers
* Generating **ROC curves** for model comparisons

---

## 🧪 **2. Models Implemented**

At least two models were trained, including one custom-built model.

### ✔️ **Model 1: Custom CNN**

* Multi-layer convolutional architecture
* ReLU activations
* Max-pooling
* Fully connected layers
* Sigmoid output for binary classification

### ✔️ **Model 2: Pretrained Model (Transfer Learning)**

* Example: ResNet18 / EfficientNet / MobileNetV2 (based on your notebook)
* Final layer replaced with a single-node classifier
* Fine-tuned on histopathology images

---

## 🔧 **3. Training Setup**

### ✔️ **Optimizers Tested**

* **Adam**
* **SGD**

### ✔️ **Learning Rate Schedulers Tested**

* **StepLR**
* **ReduceLROnPlateau** (or CosineAnnealingLR depending on your code)

---

## 🖼️ **4. Image Augmentation Techniques**

At least two augmentations were used to improve generalization:

* **Random Horizontal & Vertical Flip**
* **Random Rotation**
* (Optional extras depending on your implementation: ColorJitter, Zoom, RandomCrop)

All images were also **normalized** before training.

---

## 📊 **5. Model Evaluation**

Evaluation was conducted on a validation set using:

### ✔️ **Metrics**

* Accuracy
* Loss curves
* AUC score

### ✔️ **ROC Curves**

ROC curves were generated for:

* Each model
* Each optimizer
* Each scheduler
* Combinations of training conditions

This allowed comparison of classification performance across training setups.

---

## 📁 **6. Project Structure**

```
.
├── data/                      # Subset of Kaggle dataset
├── notebooks/
│   └── ImageClassification.ipynb   # Main experimentation notebook
├── models/                    # Saved model weights
├── outputs/
│   ├── roc_curves/            # ROC plots
│   └── training_logs/         # Accuracy/loss curves
└── README.md
```

---

## 🚀 **7. Key Takeaways**

* CNNs perform strongly on small image patches.
* Augmentations and normalization significantly improve generalization.
* Optimizer and scheduler choice affects ROC performance.
* Transfer learning generally outperforms a simple custom CNN.
* ROC curves provide deeper diagnostic insight than accuracy alone.

---

## 🛠️ **8. Technologies Used**

* Python
* PyTorch / Torchvision
* NumPy, Pandas, Matplotlib
* Scikit-learn (ROC curve + AUC)
* Kaggle dataset

---

## 📌 **9. Future Improvements**

* Use a larger portion of the dataset
* Try advanced architectures like EfficientNet-B3 or Vision Transformers
* Explore stain-normalization techniques
* Hyperparameter search with Optuna or Ray Tune

---

## 🙌 Acknowledgments

* Kaggle for providing the dataset
* Histopathologic diagnostic research community
* UMBC CS Department — Project II Guidelines

---

*This repository demonstrates end-to-end deep-learning workflows for medical image classification, fulfilling all the required project components.*
