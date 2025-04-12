# 🔍 Sonar Rock vs Mine Prediction

This is a machine learning project that uses **Logistic Regression** to classify sonar signals as either **rock** or **mine** based on 60 features extracted from sonar returns.

## 📂 Dataset Info

- **Source**: [kaggle - sonar-rock-vs-mine-prediction](https://www.kaggle.com/code/manishkr1754/sonar-rock-vs-mine-prediction)
- **Rows**: 208
- **Columns**: 60 numeric features + 1 label (R/M)
- **Label**:
  - `R` = Rock
  - `M` = Mine

## 🧠 Project Workflow

1. Data Loading with Pandas
2. Data Exploration & Grouping
3. Feature & Label Splitting
4. Train-Test Split using `train_test_split` with stratification
5. Model Training using **Logistic Regression**
6. Accuracy Evaluation on both Train & Test Data
7. Prediction System for a New Sample

## 🚀 Technologies Used

- Python
- NumPy
- Pandas
- scikit-learn
- VS Code

## 📈 Model Evaluation

- **Training Accuracy**: ~85%  
- **Testing Accuracy**: ~90%

## 🔮 Prediction Example

```python
input_data = (0.0286, 0.0453, ..., 0.0062)
prediction = model.predict(input_data_reshaped)

# Output:
# "The object is a Rock"
