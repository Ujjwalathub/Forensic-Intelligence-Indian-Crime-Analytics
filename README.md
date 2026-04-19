# 🛡️ Forensic Intelligence: Indian Crime Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![UI](https://img.shields.io/badge/Dashboard-Streamlit%20%7C%20Tailwind-success)
![Data](https://img.shields.io/badge/Dataset-Indian%20Crime%20(2001--2014)-lightgrey)

## 📌 Overview
This project shifts law enforcement data analysis from a **reactive** approach to a **proactive** Early Warning System (EWS). Using 14 years of comprehensive Indian criminal justice data, this system utilizes Machine Learning to identify hidden systemic patterns, forecast risk, and provide actionable intervention recommendations for policymakers.

## ✨ Key Predictive Models

1. **🧑‍⚖️ Juvenile Recidivism Classification**
   * **Goal:** Predict the likelihood of a juvenile becoming a repeat offender based on socio-economic, educational, and family backgrounds.
   * **Algorithm:** Gradient Boosting & Random Forest (83% Accuracy).
   * **Impact:** Identifies specific states and demographic profiles requiring immediate educational and economic intervention.

2. **🎯 Victim Vulnerability Profiling**
   * **Goal:** Track and forecast dynamic demographic shifts to identify which specific age/gender groups are becoming high-risk targets for violent crimes.
   * **Methodology:** Temporal risk scoring and Spatiotemporal Trend Analysis.
   * **Impact:** Allows police to shift protective resources dynamically rather than relying on outdated historical assumptions.

3. **🏢 Institutional Stress Early Warning System (EWS)**
   * **Goal:** Predict the risk of severe human rights violations (e.g., custodial deaths, illegal detentions) before they happen.
   * **Algorithm:** Logistic Regression with ROC-AUC validation (98.9% Accuracy).
   * **Impact:** Uses leading indicators (minor complaints, failure to take action) to flag specific state police departments that require immediate internal oversight.

## 🛠️ Tech Stack
* **Data Processing & EDA:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `joblib`
* **Visualization:** `matplotlib`, `seaborn`
* **Frontend Dashboard:** HTML/Tailwind CSS (UI Prototypes) & `Streamlit` (Live Python Inference App)

## 📁 Project Structure

```text
├── data/                               # Raw datasets (Not uploaded to Git due to size)
├── models/                             # Saved .pkl files (Random Forest, Logistic Regression)
├── Output/                             # Generated graphs (Feature importance, ROC curves)
├── 1_train_models.py                   # Master script to clean data, train models, and save them
├── 2_web_app.py                        # Streamlit inference dashboard
├── ui_templates/                       # HTML/Tailwind Glassmorphic UI designs
│   ├── home_overview.html
│   ├── juvenile_recidivism_dashboard.html
│   ├── victim_vulnerability_dashboard.html
│   └── institutional_stress_ews_dashboard.html
├── requirements.txt                    # Python dependencies
└── README.md🚀 Installation & Usage
1. Clone the Repository
Bash
git clone [https://github.com/yourusername/Forensic-Intelligence-Indian-Crime-Analytics.git](https://github.com/yourusername/Forensic-Intelligence-Indian-Crime-Analytics.git)
cd Forensic-Intelligence-Indian-Crime-Analytics
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Train the Models (Run Once)
Before running the dashboard, you must train the ML models and save their states. Make sure your datasets are placed in the data/ folder.

Bash
python 1_train_models.py
(This will generate the .pkl files in the /models directory and output visualizations to /Output)

4. Launch the Dashboard
To start the interactive predictive dashboard:

Bash
streamlit run 2_web_app.py
(Alternatively, view the static HTML templates by running python -m http.server 8000 and navigating to the /ui_templates folder)

📊 Sample Outputs & Dashboards
(Add your screenshots here)

Output/juvenile_recidivism_feature_importance.png - Shows which socio-economic factors drive repeat offenses.

Output/institutional_stress_risk_distribution.png - Highlights states currently at critical risk of institutional failure.

Output/victim_vulnerability_heatmap.png - Visualizes the intersection of crime types and vulnerable age groups.

💾 Data Source
The dataset used in this project contains complete information about various aspects of crimes that happened in India from 2001 to 2014, including police disposal, court trials, human rights violations, and juvenile backgrounds.

🤝 Contributing
Contributions, issues, and feature requests are welcome!

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
Distributed under the MIT License. See LICENSE for more information.
