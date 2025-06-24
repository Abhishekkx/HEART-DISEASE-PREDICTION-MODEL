# ❤️ **Heart Disease Prediction System**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1snzfRGRjCBCQvUzmHqDIQCcqA-BAxSSq?usp=sharing)

[📁 **Download Data & Model**](https://drive.google.com/drive/folders/1t85LaIVS4cl4833Zk239bP2Xty6_IUUw?usp=drive_link)

---

## 🫀 **What is this project?**

This is an **Machine Learning Heart Disease Prediction Model** that uses deep learning to assess cardiovascular risk.

✅ Combines data from:

* UCI Heart Disease
* Framingham Heart Study
* Cardio Train dataset

✅ Features:

* Single & batch predictions
* Visual analytics
* Training reports
* Web interface (Flask app)

---

## 🚀 **How to Run**

1️⃣ **Clone or download this repository**
2️⃣ Ensure these files exist in the directory:

* `app.py`
* `heart_disease_model.h5`
* `scaler.joblib`
* `index.html`
* All files in `static/`

3️⃣ **Install required libraries**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:

```bash
pip install flask flask-cors pandas numpy tensorflow scikit-learn joblib matplotlib seaborn
```

4️⃣ **Run the app**

```bash
python app.py
```

👉 Visit: [http://localhost:5000](http://localhost:5000)

---

## 🔁 **How to Train the Model Again**

1️⃣ Ensure you have the datasets:

* `heart_disease_uci.csv`
* `framingham.csv`
* `cardio_train.csv`

2️⃣ Install dependencies (see above)

3️⃣ Run:

```bash
python heart_disease_full_code_h5model.py
```

✅ This will:

* Preprocess data
* Train the model
* Save `heart_disease_model.h5` & `scaler.joblib`

---

## 📚 **Requirements**

* Python 3.7+
* Flask
* Flask-CORS
* pandas
* numpy
* tensorflow
* scikit-learn
* joblib
* matplotlib
* seaborn

---

## ✨ **Features**

* ⚡ Real-time heart disease risk prediction
* 📊 Interactive web interface
* 📈 Visual analytics + training reports
* 🔀 Multi-dataset integration for robust results

---

## 🔗 **Links**

* [🌐 Colab Notebook (Training + Analysis)](https://colab.research.google.com/drive/1snzfRGRjCBCQvUzmHqDIQCcqA-BAxSSq?usp=sharing)
* [📁 Google Drive (Data + Models)](https://drive.google.com/drive/folders/1t85LaIVS4cl4833Zk239bP2Xty6_IUUw?usp=drive_link)

---

### 💬 **For issues**

Please refer to the code or contact the maintainer.

