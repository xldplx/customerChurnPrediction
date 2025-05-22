# alden.

website code. i'm a web developer.

💻 **Tech Stack:**
- Styling usually done with **Tailwind CSS**.
- Front-end stuff with **React**.
- Sometimes use **Next.js** for deployment.
- Might look into other styling frameworks later.

🚀 **Projects:**
Check out the folders for my projects. If you need something specific, just ask!

🔧 **Quick Links:**
- [Frontend](/frontend)
- [Backend](/backend)

Let's build something cool together! 🎨

## Features

- **Data Pipeline**  
  - Handles missing values (median imputation)  
  - Balances data using SMOTE (85:15 → 50:50)  
  - Encodes categories with OneHotEncoder  

- **ML Model**  
  - XGBoost (optimized via GridSearchCV)  
  - Accuracy: 82.3% | Recall (Churn): 78.5%  
  - Key features: `tenure`, `MonthlyCharges`, `Contract`  

- **Web Interface**  
  - Next.js form with real-time validation  
  - Dark theme UI  


## Setup

1. **Backend**  
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

2. **Frontend**  
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Why We Used

- **XGBoost**: Best performance for imbalanced data  
- **SMOTE**: Preserves original data distribution  
- **Flask+Next.js**: Lightweight and scalable  

## Future Work

- Add SHAP explainability  
- Docker deployment  
