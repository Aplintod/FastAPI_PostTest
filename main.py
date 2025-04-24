from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import logging

# Inisialisasi FastAPI
app = FastAPI(title="Analisis Inflasi Harga Pangan Prediction API (Random Forest & XGBoost)")

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)

# Load Random Forest model dari file pickle
try:
    with open("random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    logging.info("✅ Model Random Forest berhasil dimuat.")
except Exception as e:
    logging.error(f"❌ Gagal memuat model Random Forest: {e}")
    rf_model = None

# Load XGBoost model dari file pickle
try:
    with open("xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    logging.info("✅ Model XGBoost berhasil dimuat.")
except Exception as e:
    logging.error(f"❌ Gagal memuat model XGBoost: {e}")
    xgb_model = None

# Load StandardScaler dari file pickle
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logging.info("✅ Scaler berhasil dimuat.")
except Exception as e:
    logging.error(f"❌ Gagal memuat scaler: {e}")
    scaler = None

# Input schema sesuai nama kolom saat training
class InputData(BaseModel):
    Open: float
    High: float
    Low: float
    Close: float
    Inflation: float
    avg_price: float
    year: int
    month: int
    quarter: int
    volatility: float

# Output schema untuk hasil prediksi
class PredictionResponse(BaseModel):
    prediction: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "✅ XGBoost and Random Forest API for Analisis Inflasi Harga Pangan Prediction is running"}

# Preprocessing input
def preprocess_input(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    logging.info(f"Input data: {df}")
    
    # Hapus kolom Close karena tidak digunakan dalam model
    if 'Close' in df.columns:
        df = df.drop('Close', axis=1)
        logging.info("Kolom 'Close' dihapus karena tidak digunakan dalam model")
    
    # Definisikan ulang urutan kolom yang diharapkan (sesuai dengan urutan saat training)
    expected_columns = ['Open', 'High', 'Low', 'Inflation', 'avg_price', 'year', 'month', 'quarter', 'volatility']
    
    # Periksa apakah ada kolom yang hilang
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {', '.join(missing_cols)}")
    
    # Pastikan urutan kolom sama dengan saat training
    df = df[expected_columns]
    
    if scaler is None:
        raise ValueError("Scaler belum dimuat. Pastikan file 'scaler.pkl' tersedia.")
    
    # Apply scaling normalization (using pre-loaded scaler)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=expected_columns)
    
    logging.info(f"Normalized input data: {df_scaled}")
    
    return df_scaled

# Prediction endpoint untuk Random Forest
@app.post("/predict_rf/", response_model=PredictionResponse)
def predict_rf(data: InputData):
    try:
        if rf_model is None:
            raise ValueError("Model Random Forest belum dimuat. Pastikan file 'random_forest_model.pkl' tersedia.")
        
        # Preprocess dan normalisasi input data
        df = pd.DataFrame([data.dict()])
        
        # Hapus kolom Close
        if 'Close' in df.columns:
            df = df.drop('Close', axis=1)
            logging.info("Kolom 'Close' dihapus karena tidak digunakan dalam model")
        
        # Pastikan kolom dalam urutan yang tepat (mungkin berbeda untuk model RF)
        expected_columns = ['Open', 'High', 'Low', 'Inflation', 'avg_price', 'year', 'month', 'quarter', 'volatility']
        # Periksa kolom yang hilang
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {', '.join(missing_cols)}")
        
        # Pastikan urutan kolom sama
        df = df[expected_columns]
        
        # Scaling tanpa menggunakan nama kolom
        if scaler is None:
            raise ValueError("Scaler belum dimuat. Pastikan file 'scaler.pkl' tersedia.")
        
        # Transform tanpa mempertahankan nama kolom
        df_scaled_array = scaler.transform(df)
        
        # Prediksi langsung dengan array (tanpa konversi kembali ke DataFrame)
        prediction = rf_model.predict(df_scaled_array)[0]
        
        return {"prediction": round(float(prediction), 2)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Prediction endpoint untuk XGBoost
@app.post("/predict_xgb/", response_model=PredictionResponse)
def predict_xgb(data: InputData):
    try:
        if xgb_model is None:
            raise ValueError("Model XGBoost belum dimuat. Pastikan file 'xgboost_model.pkl' tersedia.")
        
        # Preprocess dan normalisasi input data
        processed = preprocess_input(data)
        
        # Prediksi menggunakan XGBoost model
        prediction = xgb_model.predict(processed)[0]
        
        # Pastikan prediksi dikembalikan dalam format response yang sesuai
        return {"prediction": round(float(prediction), 2)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        return {"error": str(e)}
