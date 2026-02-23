# 🔊 SONAR Mine vs Rock Prediction

A Machine Learning project that predicts whether a SONAR signal is bouncing off a **Mine** or a **Rock** using Logistic Regression.

---

## 📌 Project Overview

SONAR (Sound Navigation and Ranging) sends sound waves and detects the objects based on the returning signals. This project uses those signal readings to classify underwater objects as either a **Mine (M)** or a **Rock (R)** using a supervised machine learning model.

---

## 📂 Dataset

- **Source:** [UCI Machine Learning Repository — SONAR Dataset](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks))
- **Rows:** 208 samples
- **Columns:** 61 (60 features + 1 label)
- **Features:** 60 numerical values representing sonar signal energy at different angles
- **Label:** `M` → Mine &nbsp;|&nbsp; `R` → Rock

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python | Programming Language |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Scikit-learn | ML model, train-test split, evaluation |
| Google Colab | Development environment |

---

## 🔁 Project Workflow

```
Load Dataset
     ↓
Exploratory Data Analysis (EDA)
     ↓
Separate Features (X) and Label (Y)
     ↓
Train-Test Split (90% train, 10% test)
     ↓
Train Logistic Regression Model
     ↓
Evaluate Accuracy
     ↓
Predict on New Input
```

---

## 📊 Model Performance

| Dataset | Accuracy |
|---|---|
| Training Data | ~83.4% |
| Testing Data | ~76.2% |

---

## 🧠 Key Concepts Used

- **Binary Classification** — Mine vs Rock
- **EDA** — shape, describe, value_counts
- **Stratified Train-Test Split** — balanced label distribution
- **Logistic Regression** — supervised classification algorithm
- **Prediction Pipeline** — reshape → predict → interpret result

---

## 📁 Project Structure

```
📦 01 - SONAR Mine vs Rock Prediction
 ┣ 📓 mine_vs_rock_prediction.ipynb   ← Main notebook
 ┣ 📄 sonar data.csv                  ← Dataset
 ┗ 📄 README.md                       ← Project documentation
```

---

## ▶️ How to Run

**Option 1 — Google Colab (Recommended)**
1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/)
2. Upload the `sonar data.csv` dataset
3. Run all cells from top to bottom

**Option 2 — Local Machine**
```bash
# Clone the repository
git clone https://github.com/alagarsamy-m/Machine-Learning.git

# Install required libraries
pip install numpy pandas scikit-learn

# Open the notebook
jupyter notebook mine_vs_rock_prediction.ipynb
```

---

## 🔍 Sample Prediction

```python
# Give 60 sonar signal values as input
input_data = (0.0307, 0.0523, 0.0653, ...)   # 60 values

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")
```

---

## 📈 Future Improvements

- Add StandardScaler for feature scaling to improve accuracy
- Try advanced models like SVM, Random Forest, XGBoost
- Use Cross Validation for more reliable evaluation
- Deploy the model using Streamlit or Flask

---

## 👨‍💻 Author

**Eswaran** — Final Year IT Student | Aspiring Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/alagarsamy-m)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
