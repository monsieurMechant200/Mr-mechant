"""
Neural Credit - API Backend FastAPI
Auteur: David Meilleur
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import joblib
import numpy as np
import jwt
import hashlib
import json

# Configuration
SECRET_KEY = "neural-credit-secret-key-2025-cameroun"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 heures

# Initialisation FastAPI
app = FastAPI(
    title="Neural Credit API",
    description="API de scoring alternatif basée sur l'IA pour le Cameroun",
    version="1.0.0"
)

# CORS (pour le dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle au démarrage
try:
    MODEL = joblib.load('models/neural_credit_model.pkl')
    SCALER = joblib.load('models/scaler.pkl')
    LABEL_ENCODERS = joblib.load('models/label_encoders.pkl')
    
    with open('models/feature_names.json', 'r') as f:
        FEATURE_NAMES = json.load(f)
    
    with open('models/model_metadata.json', 'r') as f:
        MODEL_METADATA = json.load(f)
    
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"⚠️ Erreur lors du chargement du modèle : {e}")
    MODEL = None

# Modèles Pydantic
class UserCredentials(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Âge de l'utilisateur")
    gender: str = Field(..., pattern="^(M|F)$", description="Genre (M/F)")
    region: str = Field(..., description="Région au Cameroun")
    profession: str = Field(..., description="Profession")
    monthly_income: float = Field(..., gt=0, description="Revenu mensuel en FCFA")
    
    # Données comportementales
    geo_stability: float = Field(..., ge=0, le=1, description="Stabilité géographique")
    calls_per_week: int = Field(..., ge=0, description="Appels hebdomadaires")
    habit_consistency: float = Field(..., ge=0, le=1, description="Cohérence des habitudes")
    psy_score: float = Field(..., ge=0, le=1, description="Score psychométrique")
    social_activity: int = Field(..., ge=0, description="Posts sociaux par semaine")
    
    # Données transactionnelles
    tx_avg_amount: float = Field(..., ge=0, description="Montant moyen des transactions")
    tx_count: int = Field(..., ge=0, description="Nombre total de transactions")
    tx_frequency_monthly: float = Field(..., ge=0, description="Fréquence mensuelle")

class ScoringResponse(BaseModel):
    user_id: Optional[str] = None
    neural_trust_score: float = Field(..., description="Score de confiance (0-100)")
    risk_category: str = Field(..., description="Catégorie de risque")
    recommended_loan_amount: float = Field(..., description="Montant recommandé (FCFA)")
    recommended_interest_rate: float = Field(..., description="Taux d'intérêt recommandé (%)")
    explanation: Dict[str, any] = Field(..., description="Facteurs explicatifs")
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: float

# Sécurité OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Base de données utilisateurs simulée (à remplacer par PostgreSQL en production)
FAKE_USERS_DB = {
    "david@neuralcredit.cm": {
        "email": "david@neuralcredit.cm",
        "hashed_password": hashlib.sha256("password123".encode()).hexdigest(),
        "role": "admin"
    },
    "demo@neuralcredit.cm": {
        "email": "demo@neuralcredit.cm",
        "hashed_password": hashlib.sha256("demo123".encode()).hexdigest(),
        "role": "institution"
    }
}

# Fonctions utilitaires
def create_access_token(data: dict):
    """Crée un JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Vérifie un JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Récupère l'utilisateur courant"""
    email = verify_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide ou expiré",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = FAKE_USERS_DB.get(email)
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return user

def calculate_risk_category(score: float) -> str:
    """Détermine la catégorie de risque"""
    if score >= 75:
        return "Faible risque (A)"
    elif
