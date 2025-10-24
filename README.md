# 🌾 Pirinç Türlerini Sınıflandırma CNN Projesi

# 🌾 Rice Variety Classification with CNN

## 📘 Proje Hakkında / Project Overview

Bu proje, farklı pirinç türlerini sınıflandırmak için Convolutional Neural Network (CNN) tabanlı bir derin öğrenme modeli geliştirmeyi amaçlamaktadır. Veri seti Kaggle’dan alınmıştır ve model pirinç görsellerini kullanarak türlerini otomatik olarak tanımayı öğrenmektedir.

This project aims to develop a Convolutional Neural Network (CNN)-based deep learning model to classify different rice varieties. The dataset was obtained from Kaggle, and the model learns to automatically recognize rice types from images.

## 🔍 Amaç / Objective

* Pirinç türlerini yüksek doğrulukla sınıflandırabilen bir model geliştirmek

* Görüntü tabanlı veri analizi ve derin öğrenme pratiklerini uygulamak

* To build a model that can classify rice varieties with high accuracy

* To apply image-based data analysis and deep learning practices

## 📊 Veri Seti / Dataset

Veri seti, Kaggle üzerinde bulunan pirinç türleri görsellerinden oluşmaktadır. Her bir görsel, bir pirinç türünü temsil eder ve model bu görseller üzerinden sınıflandırma yapar.

The dataset consists of images of rice varieties available on Kaggle. Each image represents a rice type, and the model classifies based on these images.

## 🧪 Yöntem / Methodology

* Convolutional Neural Network (CNN) kullanılarak model geliştirilmiştir

* Veri ön işleme: Görseller boyutlandırma, normalizasyon ve veri artırma (augmentation)

* Eğitim ve test veri setlerinin oluşturulması

* Modelin eğitilmesi ve doğruluk skorlarının ölçülmesi

* The model is developed using a Convolutional Neural Network (CNN)

* Data preprocessing: resizing, normalization, and data augmentation

* Creating training and testing datasets

* Training the model and evaluating accuracy scores

## 📈 Sonuçlar / Results

Model, test veri seti üzerinde yüksek doğruluk göstermiştir ve pirinç türlerini başarıyla sınıflandırabilmektedir.

The model demonstrated high accuracy on the test dataset and can successfully classify rice varieties.

## 📂 Dosya Yapısı / File Structure

```
CNN-Deep-Learning/
│
├── dataset/               # Kaggle'dan alınan pirinç görselleri / Rice images from Kaggle
├── cnn_model.py           # CNN modelinin tanımı ve eğitimi / CNN model definition and training
├── preprocess.py          # Veri ön işleme kodları / Data preprocessing scripts
├── requirements.txt       # Gerekli Python kütüphaneleri / Required Python libraries
└── README.md              # Proje hakkında bilgi / Project information
```

## 🛠️ Kurulum / Installation

Gerekli Python kütüphanelerini yüklemek için:

To install the required Python libraries:

```bash
pip install -r requirements.txt
```
