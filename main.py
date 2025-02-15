import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """Load and clean the basketball dataset, calculating eFG%."""
    df = pd.read_csv(file_path, sep=',', on_bad_lines='skip', engine='python', header=0)

    # Print column names for debugging
    print("Column Names in Dataset:", df.columns.tolist())

    # Ensure required columns exist
    required_cols = {'FGM_2', 'FGA_2', 'FGM_3', 'FGA_3', 'FTA', 'FTM', 'AST', 'BLK', 
                     'STL', 'TOV', 'TOV_team', 'DREB', 'OREB', 'F_tech', 'F_personal',
                     'largest_lead', 'rest_days', 'tz_dif_H_E', 'prev_game_dist',
                     'home_away_NS', 'travel_dist', 'team_score', 'opponent_team_score'}
    
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns in dataset: {missing_cols}")

    # Compute Effective Field Goal Percentage (eFG%)
    df['eFG%'] = (df['FGM_2'] + df['FGM_3'] * 1.5) / (df['FGA_2'] + df['FGA_3'])
    
    # Create the actual winner label (1 if team won, 0 otherwise)
    df['actual_win'] = (df['team_score'] > df['opponent_team_score']).astype(int)

    return df

def preprocess_data(df, features):
    """Preprocess data: handle missing values, split into train and test sets."""
    df_clean = df.dropna(subset=features + ['actual_win'])  # Remove rows with missing values
    
    if df_clean.empty:
        raise ValueError("Dataframe is empty after dropping missing values. Check dataset.")

    X = df_clean[features]
    y = df_clean['actual_win']  # Label based on team performance, not scores

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate a Logistic Regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train and evaluate an XGBoost model."""
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, y_pred

if __name__ == "__main__":
    # File path (Ensure basketball_data.csv is in the same directory)
    file_path = "basketball_data.csv"

    # Features used in the model (Now excludes team_score and opponent_team_score)
    features = ['eFG%', 'FTA', 'FTM', 'AST', 'BLK', 'STL', 'TOV',
                'TOV_team', 'DREB', 'OREB', 'F_tech', 'F_personal', 
                'largest_lead', 'rest_days', 'tz_dif_H_E', 'prev_game_dist', 
                'home_away_NS', 'travel_dist']

    # Load and preprocess data
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, features)

    # Train models and get predictions
    log_reg_acc, log_reg_pred = train_logistic_regression(X_train, X_test, y_train, y_test)
    xgb_acc, xgb_pred = train_xgboost(X_train, X_test, y_train, y_test)

    # Compare predictions with actual winners
    actual_winners = y_test.values  # Ground truth

    # Print accuracy results
    print(f"\nLogistic Regression Accuracy (Predicted vs. Actual): {log_reg_acc:.4f}")
    print(f"XGBoost Accuracy (Predicted vs. Actual): {xgb_acc:.4f}")

    print("\n📌 Formulas Used in Prediction:")
    print("1️⃣ Effective Field Goal Percentage (eFG%) = (FGM + 0.5 * 3PM) / FGA")
    print("2️⃣ Actual Win = 1 if team_score > opponent_team_score, else 0 (Used for evaluation)")
    print("3️⃣ Standardization: X_scaled = (X - mean) / std_dev")