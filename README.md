# ML-PROJECT-
# Customer Retail Transaction Classification Using Machine Learning

A supervised ML project to classify retail transactions as High-Value or Low-Value using Logistic Regression, Decision Tree, and K-Nearest Neighbors.

## 📊 Project Overview
This project applies binary classification algorithms on the `customer_retail.csv` dataset to predict whether a transaction is High-Value based on `Quantity`, `UnitPrice`, and `Country`. The key objective is to compare model performance on imbalanced data and identify issues like data leakage.

## 📁 Files in this Repository
| File | Description |
| --- | --- |
| `ML_Project.ipynb` | Jupyter notebook with all code blocks and outputs |
| `customer_retail.csv` | Original dataset - 397,884 transactions |
| `ML_Project_.pdf` | Final project report with analysis and results |

## 🛠️ Technologies Used
- **Language**: Python 3
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn
- **Algorithms**: Logistic Regression, Decision Tree, KNN

## 📈 Key Results
| Model | Accuracy | Key Finding |
| --- | --- | --- |
| Logistic Regression | 85.07% | Low recall 0.41 for High-Value class due to imbalance |
| Decision Tree | 100.00% | Overfitting caused by data leakage from target engineering |
| K-Nearest Neighbors | 99.90% | Near-perfect due to same leakage issue |

**Main Insight**: The Decision Tree achieved 100% accuracy because it learned the exact rule `TotalPrice = Quantity * UnitPrice > 20` used to create the target variable. Logistic Regression is the only generalizable model from this experiment.

## 🚀 How to Run
1. Clone this repo: `git clone https://github.com/your-username/customer-retail-ml-classification.git`
2. Install dependencies: `pip install pandas numpy matplotlib scikit-learn jupyter`
3. Open `ML_Project.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells sequentially

## 📊 Dataset
- **Source**: `customer_retail.csv` 
- **Size**: 397,884 transactions after cleaning
- **Features**: `Quantity`, `UnitPrice`, `Country`, `InvoiceNo`, `CustomerID`
- **Target**: `HighValue` — engineered as 1 if `Quantity * UnitPrice > 20`, else 0
- **Class Distribution**: 62,527 Low-Value vs 17,050 High-Value transactions

## 🔍 Future Improvements
1. Use independent features like `InvoiceDate` seasonality instead of derived `TotalPrice`
2. Handle class imbalance with SMOTE or class weighting
3. Test on true hold-out data to verify generalization

## 👤 Author
**arakkalamna**  
Machine Learning Project | 2026
