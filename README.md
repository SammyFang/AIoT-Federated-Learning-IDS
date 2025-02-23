# AIoT-Federated-Learning-IDS
Heterogeneous Data-Aware Federated Learning for Intrusion Detection Systems A privacy-preserving federated learning (FL) framework for intrusion detection (IDS) in AIoT environments, leveraging meta-sampling, dynamic clustering, and FedProx optimization to handle class imbalance and non-IID data distributions.

![image](https://github.com/user-attachments/assets/6146160b-e351-4f2d-9e9c-e6139216743c)

# **Heterogeneous Data-Aware Federated Learning for Intrusion Detection Systems via Meta-Sampling in AIoT**

This project implements a **federated learning (FL) based intrusion detection system (IDS)** tailored for **Artificial Intelligence of Things (AIoT)** environments. The proposed **Clustering-Enabled Federated Meta-Training (CFMT) framework** overcomes **class imbalance and non-IID data distribution challenges** in FL-based IDS by incorporating **meta-sampling and dynamic clustering** techniques.

## 📌 **Key Features**
- **Federated Learning for IDS**: Privacy-preserving model training across AIoT devices.
- **Meta-Sampler for Class Balancing**: Adaptive sampling improves model generalization.
- **Dynamic Clustering Algorithm**: Groups clients based on data characteristics to enhance aggregation.
- **Federated Proximal Optimization**: Reduces the impact of heterogeneous data distributions.
- **Multi-Layer Perceptron (MLP) Model**: Optimized for intrusion detection tasks.

---

## 🏗 **System Architecture**
The **CFMT Framework** operates in a federated learning setting and consists of the following components:

### **1️⃣ Meta-Sampler**
- Dynamically balances local datasets using a Gaussian-based weighting function.
- Constructs meta-states from training and validation error distributions.
  
### **2️⃣ Dynamic Clustering**
- Groups client models based on auxiliary vectors (class imbalance ratio, accuracy).
- Ensures effective aggregation of local models despite **non-IID data distributions**.

### **3️⃣ Federated Learning Algorithm**
- Uses **FedProx** for global model aggregation while mitigating data heterogeneity.

---

## 🔥 **Implementation Details**
### **🖥️ Model Architecture**
The core model is a **Multi-Layer Perceptron (MLP)** with:
- **Input Layer**: Matches dataset feature dimensions.
- **Hidden Layer**: 128 neurons with **ReLU activation**.
- **Output Layer**: Binary classification for intrusion detection.

## 📊 **Experiment Setup**
### **📂 Dataset**
NSL-KDD: Standard benchmark dataset for intrusion detection systems.
Data partitioning techniques:
Class Imbalance Partitioning: Simulates real-world attack prevalence.
Non-IID Partitioning: Uses Dirichlet distribution for heterogeneous data distributions.

## 🏆**Baseline Comparisons**
- **Method	F1-Score	AUC**
- **FedAvg	0.78	0.84**
- **FedProx	0.80	0.85**
- **Fed-Over	0.82	0.87**
- **Fed-Under	0.79	0.84**
- **CFMT (Proposed)	0.86	0.90**

## 🏆**Training Curves**
The CFMT framework consistently outperformed baseline FL methods by effectively handling class imbalance and non-IID data, resulting in higher F1-Scores and AUC values.

![image](https://github.com/user-attachments/assets/219368c8-bccc-4325-93f1-addd768e603e)
![image](https://github.com/user-attachments/assets/621f7797-052e-41c9-8950-d01157ae772f)

## **🔬 Results & Discussion**
### **💡 Handling Class Imbalance**
- Baseline methods (FedAvg, FedProx) degrade in performance under severe class imbalance.
- CFMT meta-sampling ensures the model is exposed to balanced training samples, improving generalization.
## **📊 Performance with Non-IID Data**
- The dynamic clustering approach mitigates the effects of non-IID distributions.
- The proposed framework achieves higher accuracy and more stable training convergence than baseline models.
