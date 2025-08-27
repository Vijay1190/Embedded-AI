# Bioacoustic Insect Classification on STM32 (Edge AI)

This project develops a **Compact CNN-based Edge AI System** for **Bioacoustic Insect Classification** on resource-constrained STM32 microcontrollers.  
It extends from a **Stage 1 proof-of-concept** (motion classification using accelerometer data on STM32F407VGT6) to a more advanced **Stage 2 project** (insect sound classification using 2D CNNs on STM32F769I).  

---

## ✅ Stage 1 — Accomplished (Motion Classification)

**Topic**: Motion Classification from 3-axis accelerometer data  

- **Platform**: STM32F407VGT6 (Cortex-M4 @ 168 MHz)  
- **Model**: Lightweight 1D CNN  
- **Deployment**: Fully implemented in Embedded C  
- **Optimization**: footprint reduced to **~27KB**  
- **Classes**: Static, Walking, Running  
- **Accuracy**: 89.2%  
- **Performance**: Real-time inference verified with low latency  

This stage validated the feasibility of **on-device deep learning** on Cortex-M MCUs.  

---

## 🚀 Stage 2 — Current Research (Bioacoustic Insect Classification)

**Topic**: Edge AI for insect sound recognition using **bioacoustic signatures**  

### 🎯 Objective  
Build a **real-time 2D CNN** pipeline for detecting and classifying insect species (e.g., mosquito, bee, cricket) from **acoustic spectrograms**.  

### 🔧 Setup
- **Platform**: STM32F769I (Cortex-M7 @ 216 MHz, more compute power & memory)  
- **Input**: Audio captured from MEMS microphones  
- **Preprocessing**:  
  - STFT / Mel Spectrogram conversion  
  - Sliding window segmentation  
  - Normalization  
- **Model**:  
  - Lightweight **2D CNN / MobileNet-inspired blocks**  
  - Optimized for embedded inference  
- **Optimization**:   
  - Direct Embedded C Weights and Bias Implementation.  
  - Knowledge distillation for compactness  

### 📊 Expected Outcomes
- **Classes**: Multiple insect species  
- **Applications**: For Insect Pest Classification In Rice Stores. 
- **Deployment**: MCU-level real-time inference with <100 KB footprint  

---

## 🛠️ Tech Stack

- **Frameworks**: TensorFlow/Keras (training), Embedded C (deployment)  
- **Languages**: Python, Embedded C  
- **Hardware**:  
  - Stage 1: STM32F407VGT6 (accelerometer-based motion classification)  
  - Stage 2: STM32F769I + MEMS microphone (bioacoustic sound classification)  

---


## ✍️ Author
Developed by **Dr. VijayaKumar S**  
and Student Contributer **Dhanvanth S**

Part of an **Edge AI Embedded Systems Research Project**  

---

⭐ If you like this project, consider giving it a **star** to support research in **TinyML for ecological monitoring**!
