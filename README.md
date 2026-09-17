# 🌱 AgroMitra  
**AI-Powered Crop Disease Detection & Precision Agriculture Platform**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C99C6?style=flat-square&logo=opencv)](https://opencv.org/)


</div>

## Overview

AgroMitra combines **deep learning disease detection** with **IoT sensor fusion** to deliver actionable agricultural insights. Using fine-tuned **MobileNetV2** and **OpenCV preprocessing**, it classifies tomato diseases (Healthy, Early Blight, Late Blight) with 94.8% accuracy, then generates severity assessments and treatment recommendations based on real-time environmental data.

## ✨ Key Features

- **Multi-Class Disease Detection:** Identifies 3 tomato disease states with high confidence
- **Environmental Fusion:** Integrates temperature, humidity, and soil moisture for contextualized diagnostics
- **Intelligent Recommendations:** Prescribes fungicides, irrigation adjustments, and canopy management
- **Transfer Learning:** Lightweight MobileNetV2 for edge deployment
- **Dynamic Class Weighting:** Handles dataset imbalances automatically
- **Production-Ready:** Comprehensive preprocessing, augmentation, and error handling

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | TensorFlow/Keras |
| **Transfer Learning** | MobileNetV2 |
| **Computer Vision** | OpenCV 4.x |
| **Data Processing** | NumPy, Pandas |
| **Language** | Python 3.12 |

## 📊 System Architecture


Leaf Image → OpenCV Preprocessing → MobileNetV2 Classifier → Decision Engine
                                          ↓                        ↓
                                    Disease Class         IoT Sensor Data
                                    + Confidence          ↓
                                                    Four-Tier Report
                                                    (Classification, Confidence,
                                                     Severity, Recommendations)



## 📁 Project Structure

agromitra/
├── data/                    # Ignored via .gitignore (empty or local only)
├── models/                  
│   └── best_mobilenetv2.keras
├── src/
│   ├── decision_engine.py
│   ├── predict.py
│   └── train.py
├── main.py                  # Entry point script
└── README.md

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.8% |
| **F1-Score** | 0.944 |
| **Inference Time** | 85ms/image |

**Per-Class Performance:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Healthy | 0.97 | 0.96 | 0.96 |
| Early Blight | 0.93 | 0.94 | 0.93 |
| Late Blight | 0.94 | 0.95 | 0.94 |

## 💡 Usage Example

```python
from src.inference import AgroMitraPredictor
from src.decision_engine import DecisionEngine

predictor = AgroMitraPredictor('./models/agromitra_v1.h5')
disease, confidence = predictor.predict('./leaf.jpg')

engine = DecisionEngine()
report = engine.generate_report(
    disease_class=disease,
    confidence=confidence,
    sensor_data={'temperature': 20.5, 'humidity': 82, 'soil_moisture': 68}
)
print(report)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with ❤️ for precision agriculture**


</div>
