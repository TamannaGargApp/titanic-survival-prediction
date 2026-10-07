# Titanic Survival Prediction 🚢

A beginner-friendly **data preprocessing + machine learning** project on the classic Kaggle Titanic dataset.
The goal: given information about a passenger, predict whether they **survived** or not.

The project covers the full workflow: inspecting raw data, handling missing values, encoding categorical features, **comparing three models** (Logistic Regression, Decision Tree, Random Forest), studying **overfitting**, and saving the best model. There's also a bonus **K-Means clustering** section that groups passengers without using the survival labels.

---

## 📁 Project structure

```
titanic-survival-prediction/
├── data/                  # put train.csv here (not tracked by git)
│   └── README.md
├── models/                # the trained model is saved here (not tracked by git)
├── notebooks/
│   └── titanic_preprocessing_and_model.ipynb   # step-by-step walkthrough
├── src/
│   └── train.py           # same pipeline as a runnable script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📥 Getting the data

The dataset belongs to Kaggle, so it is **not included** in this repository. Download it yourself using any one of the options below, then place `train.csv` inside the `data/` folder.

### Option 1 — Download from the website (easiest)

1. Create a free account at [kaggle.com](https://www.kaggle.com) and sign in.
2. Open the Titanic competition: <https://www.kaggle.com/competitions/titanic/data>
3. Click **Join Competition** and accept the rules (required before downloading).
4. Click **Download All**, unzip it, and copy `train.csv` into `data/`.

### Option 2 — Kaggle API (command line)

```bash
pip install kaggle
```

1. On Kaggle go to **Settings → API → Create New Token**. This downloads `kaggle.json`.
2. Move it to the right place:
   - Linux / macOS: `~/.kaggle/kaggle.json` (then run `chmod 600 ~/.kaggle/kaggle.json`)
   - Windows: `C:\Users\<your-username>\.kaggle\kaggle.json`
3. Accept the competition rules once on the website (step 3 above), then run:

```bash
kaggle competitions download -c titanic -p data
cd data && unzip titanic.zip && cd ..
```

### Option 3 — Run it directly on Kaggle

Upload the notebook to Kaggle (**Create → New Notebook → File → Import Notebook**) and add the Titanic competition data via **Add Input**. The notebook automatically uses the Kaggle path `/kaggle/input/competitions/titanic/train.csv` when the local file isn't found.

After downloading, you should have:

```
data/
├── train.csv   # 891 passengers, includes the Survived label  ← used here
├── test.csv    # 418 passengers, no Survived label (for Kaggle submissions)
└── gender_submission.csv
```

> ⚠️ Never commit the CSV files — `.gitignore` already excludes them.

---

## ⚙️ Setup & run

```bash
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run the notebook**

```bash
jupyter notebook notebooks/titanic_preprocessing_and_model.ipynb
```

**Or run the script**

```bash
python src/train.py
# or point it at a different file:
python src/train.py --data path/to/train.csv
```

The script prints a comparison table of all three models (test accuracy + 5-fold cross-validation), then saves the final Random Forest to `models/titanic_random_forest.joblib`.

---

## 🧹 Data preprocessing

The raw `train.csv` has **891 rows × 12 columns**. Kaggle provides separate `train.csv` and `test.csv`, but `test.csv` has no `Survived` labels, so `train.csv` is cleaned and then split 80/20 into training and test sets to measure accuracy.

| Step | What was done | Why |
|---|---|---|
| Drop columns | `PassengerId`, `Name`, `Ticket`, `Cabin` | IDs and free text carry no direct signal; `Cabin` is ~77% missing (687/891) |
| Fill `Age` | 177 missing → median (28.0) | Median is robust to outliers |
| Fill `Embarked` | 2 missing → mode (`S`) | Most passengers boarded at Southampton |
| Encode `Sex` | `male → 0`, `female → 1` | Models need numeric input |
| Encode `Embarked` | `S → 0`, `C → 1`, `Q → 2` | Models need numeric input |

**Final features (7):** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

---

## 🤖 Models & results

All models use the same 80/20 split (712 train / 179 test rows, `random_state=42`).

| Model | Test accuracy |
|---|---|
| Logistic Regression (baseline) | 79.9% |
| Decision Tree (`max_depth=3`) | 79.9% |
| **Random Forest** (100 trees) ✅ final model | **82.7%** |

> Run `python src/train.py` to also see **5-fold cross-validation** scores. They're a more reliable comparison than one 179-row test set, where a single passenger moves accuracy by about 0.6%.

### Final model: Random Forest

Random Forest scored highest, so it's the model that gets saved (`models/titanic_random_forest.joblib`). Logistic Regression is kept as the baseline it's measured against.

**Feature importance**

| Feature | Importance |
|---|---|
| Fare | 0.270 |
| Sex | 0.268 |
| Age | 0.256 |
| Pclass | 0.082 |
| SibSp | 0.052 |
| Parch | 0.037 |
| Embarked | 0.034 |

Tree-based importance favours columns with many distinct values (Fare, Age), and Fare overlaps heavily with Pclass. That's why Pclass ranks lower here than in Logistic Regression, where it's the second strongest signal after Sex.

### Overfitting experiment (Decision Tree)

| max_depth | Train accuracy | Test accuracy |
|---|---|---|
| 1 | 0.788 | 0.782 |
| 3 | 0.834 | 0.799 |
| 8 | 0.899 | 0.793 |
| 20 | 0.978 | 0.799 |

As the tree gets deeper, training accuracy climbs towards 98% while test accuracy stays around 80%. The tree is memorising the training data instead of learning general patterns. Random Forest reduces this by averaging many trees.

### Bonus: K-Means clustering

Passengers are grouped by **Age and Fare** (scaled) into 3 clusters without using the `Survived` column. The clusters still show clearly different survival rates, mostly separating higher-fare passengers, who survived more often, from lower-fare ones.

---

## 🚀 Future improvements

- [ ] Feature engineering: `FamilySize`, passenger title from `Name` (Mr, Mrs, Miss, Master)
- [ ] One-hot encode `Embarked` (it has no natural order)
- [ ] Tune Random Forest (`max_depth`, `n_estimators`, `min_samples_leaf`) with `GridSearchCV`
- [ ] Try Gradient Boosting / XGBoost
- [ ] Apply the same pipeline to `test.csv` and submit to Kaggle

---

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · joblib · Jupyter

## 🙏 Acknowledgements

Dataset: [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) on Kaggle.
