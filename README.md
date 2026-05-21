Fraud-anamoly-detector — Mobile Money Fraud Detection
Live demo: fraudlens.streamlit.app  |  Dataset: PaySim on Kaggle

Fraud detection sounds straightforward until you look at the numbers. In this dataset, only 0.13% of 6.3 million transactions are fraudulent. That means a model that just says "everything is legitimate" would be 99.87% accurate — and completely useless. I wanted to understand what it actually takes to find fraud in that kind of imbalance, so I built this from scratch.

What I explored:

I started with EDA before touching any model code. A few things stood out immediately.
Fraud only happens in two transaction types — TRANSFER and CASH_OUT. Every single PAYMENT, DEBIT, and CASH_IN in the dataset is legitimate. That makes sense when you think about it: those are the only two types where money can actually leave the system. So I filtered the training data down to just those two types before doing anything else.

The second thing I noticed was the balance pattern. When fraud occurs, the sender's account balance drops to exactly zero almost every single time. The fraudster takes everything and leaves nothing. In legitimate transactions, the balance hitting zero happens occasionally but nowhere near as consistently. That became the strongest feature in the model — not the transaction amount, not the time of day, but whether the origin account was completely drained.

The third thing was amount. Fraud transactions tend to be much larger than legitimate ones — the median fraud amount is around 6 to 10 times higher. Not surprising once you know fraudsters are moving as much as they can before getting caught.

How the model works

I engineered 10 features from those findings — balance drain amount, a flag for zero balance, the ratio of transaction amount to account balance, balance inconsistencies on both sides of the transaction, log-transformed amount, and a few others.

I trained three models: Logistic Regression as a baseline, Random Forest as the main model, and XGBoost as a challenger. Random Forest and Logistic Regression use class_weight='balanced' which tells the algorithm to treat each fraud case as if it appeared 750 times more often than it actually does — without this, the model just learns to predict "legitimate" for everything and scores great on accuracy while being completely useless.

XGBoost doesn't support class_weight — it uses a different parameter called scale_pos_weight, which does the same thing but requires you to calculate the ratio yourself: number of legitimate transactions divided by number of fraudulent ones. On this dataset that comes out to about 336, so XGBoost was told to weight each fraud case 336 times more heavily than a legitimate one during training.
ModelAUC-ROCAvg PrecisionFraud PrecisionFraud RecallLogistic Regression0.99120.62160.0690.960Random Forest0.99850.99711.0000.997XGBoost0.99790.99700.8720.997

Random Forest wins, and the reason is Fraud Precision. Random Forest flagged zero legitimate transactions as fraud — Fraud Precision of 1.0. XGBoost had a Fraud Precision of 0.872, meaning roughly 13% of its fraud alerts were false positives — legitimate transactions incorrectly blocked. In a real banking context, that matters. A false positive isn't just a metric — it's a customer whose payment gets declined. That trade-off is what pushed the decision toward Random Forest despite XGBoost's slightly lower AUC-ROC.

The non-linear relationships between the balance features — particularly how drain amount interacts with the account type and amount-to-balance ratio — are what tree-based models are good at capturing. The top feature by importance is balance_drain_orig, followed by orig_balance_mismatch (which catches cases where the before/after balances don't add up correctly, a sign of manipulated transactions).

One thing I was careful about: the feature engineering function is shared between the training pipeline and the Streamlit app. The exact same code runs at training time and prediction time, so there's no way for the features to be calculated differently between the two — a common bug in production ML systems called train-serve skew.

The app

I wanted the output to be something usable, not just a notebook with metrics. It has:

A transaction anomaly scatter plot — amount vs balance drain, coloured by fraud/legitimate, log-scaled on the x-axis so the full range is visible

Type breakdown charts showing volume and fraud rate per transaction type
Amount distribution comparing fraud vs legitimate transactions

A 30-day trend line showing daily fraud volume across the simulation period

An hourly pattern chart

A top-10 highest-risk table that scores a sample of the current filtered data and surfaces the most suspicious transactions

An adjustable detection threshold slider — this is the part I find most interesting from a business perspective. Setting it at 30% catches more fraud but generates more false alarms for analysts to review. Setting it at 70% means fewer alerts but real fraud cases get missed. That trade-off is a real decision fraud teams make.

A live prediction form where you can enter any transaction's details and get a fraud probability score with five risk signals explained


Running it locally
You need the PaySim dataset from Kaggle first — download it at kaggle.com/datasets/ealaxi/paysim1 and place the CSV in the data/ folder.
bashgit clone https://github.com/MadhushreeVaratharaj/fraud-anomaly-detector
cd fraud-anomaly-detector

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python notebooks/02_feature_engineering.py
python notebooks/03_train_model.py

streamlit run app/streamlit_app.py

Feature engineering takes about 2 minutes on the full 6.3M rows. Model training takes 5–10 minutes depending on your machine — the Random Forest with 300 trees on 2.2 million rows is the slow part.

Project structure

fraud-anomaly-detector/
├── data/                              ← CSV goes here (not committed to GitHub)

├── notebooks/
│   ├── 01_eda.py                      ← full EDA, generates 5 plots

│   ├── 02_feature_engineering.py      ← filter, engineer features, split

│   ├── 03_train_model.py              ← train all three models, evaluate, save

│   └── feature_engineering_utils.py  ← shared feature logic

├── models/
│   ├── fraud_model.pkl                ← saved Random Forest

│   ├── model_meta.json                ← metrics and metadata

│   └── eval_plots/                    ← ROC, PR curve, confusion matrix, feature importance, model comparison

├── app/
│   └── streamlit_app.py              ← the fraud-money-detector app

├── .streamlit/
│   └── config.toml                   ← colour theme

├── requirements.txt

└── README.md

Stack

Python · pandas · scikit-learn · XGBoost · streamlit · plotly · joblib · matplotlib · seaborn

Madhushree Varatharaj · MSc Data Science, SUTD Singapore

LinkedIn · GitHub · madhushreevaratharaj08@gmail.com