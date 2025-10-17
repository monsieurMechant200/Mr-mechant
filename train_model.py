"""
Neural Credit - Pipeline d'entraînement du modèle ML
Auteur: David Meilleur
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import json
from datetime import datetime

class NeuralCreditModel:
    """Classe principale pour le modèle de scoring"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.model_metadata = {}
        
    def load_data(self, users_path='data/users_features.csv', 
                  transactions_path='data/transactions.csv'):
        """Charge et prépare les données"""
        
        print("📥 Chargement des données...")
        users_df = pd.read_csv(users_path)
        transactions_df = pd.read_csv(transactions_path)
        
        # Feature engineering sur les transactions
        tx_features = self._extract_transaction_features(transactions_df)
        
        # Fusion avec les données utilisateurs
        df = users_df.merge(tx_features, on='user_id', how='left')
        
        # Gestion des valeurs manquantes
        df = df.fillna(0)
        
        return df
    
    def _extract_transaction_features(self, tx_df):
        """Extrait des features agrégées des transactions"""
        
        features = tx_df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'sum', 'count'],
            'tx_type': lambda x: x.value_counts().to_dict(),
            'hour': ['mean', 'std']
        }).reset_index()
        
        # Aplatir les colonnes multi-niveaux
        features.columns = ['user_id', 'tx_avg_amount', 'tx_std_amount', 
                           'tx_total_amount', 'tx_count', 'tx_types', 
                           'tx_avg_hour', 'tx_std_hour']
        
        # Features dérivées
        features['tx_frequency_monthly'] = features['tx_count'] / 6
        features['tx_stability_ratio'] = features['tx_std_amount'] / (features['tx_avg_amount'] + 1)
        
        # Nombre de dépôts vs retraits (indicateur de flux positif)
        def count_type(tx_dict, tx_type):
            return tx_dict.get(tx_type, 0) if isinstance(tx_dict, dict) else 0
        
        features['deposit_count'] = features['tx_types'].apply(lambda x: count_type(x, 'Dépôt'))
        features['withdrawal_count'] = features['tx_types'].apply(lambda x: count_type(x, 'Retrait'))
        features['balance_indicator'] = features['deposit_count'] - features['withdrawal_count']
        
        # Suppression de la colonne dict
        features = features.drop('tx_types', axis=1)
        
        return features
    
    def prepare_features(self, df):
        """Prépare les features pour l'entraînement"""
        
        # Séparation features / target
        target_col = 'is_good_payer'
        id_col = 'user_id'
        
        # Colonnes à exclure
        exclude_cols = [id_col, target_col, 'credit_score', 'registration_date']
        
        # Features numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Features catégorielles
        categorical_cols = ['gender', 'region', 'profession']
        
        # Encodage des catégorielles
        df_encoded = df.copy()
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Sélection finale des features
        feature_cols = numeric_cols + [col + '_encoded' for col in categorical_cols if col in df.columns]
        self.feature_names = feature_cols
        
        X = df_encoded[feature_cols]
        y = df[target_col]
        
        return X, y
    
    def train(self, X, y, use_xgboost=True):
        """Entraîne le modèle"""
        
        print("\n🧠 Entraînement du modèle...")
        
        # Split train/test avec stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choix du modèle
        if use_xgboost:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        
        # Entraînement
        self.model.fit(X_train_scaled, y_train)
        
        # Prédictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Métriques
        auc = roc_auc_score(y_test, y_proba)
        
        print("\n✅ Entraînement terminé !")
        print(f"\n📊 Performances sur le jeu de test :")
        print(f"   - AUC-ROC : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Mauvais payeur', 'Bon payeur'])}")
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📉 Matrice de confusion :")
        print(f"   TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"   FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Validation croisée
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        print(f"\n🔄 Validation croisée (5-fold) :")
        print(f"   - AUC moyen : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Sauvegarde des métadonnées
        self.model_metadata = {
            'model_type': 'XGBoost' if use_xgboost else 'RandomForest',
            'auc_roc': float(auc),
            'cv_auc_mean': float(cv_scores.mean()),
            'cv_auc_std': float(cv_scores.std()),
            'n_features': len(self.feature_names),
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train)
        }
        
        return self.model, auc
    
    def get_feature_importance(self, top_n=15):
        """Retourne l'importance des features"""
        
        if self.model is None:
            raise ValueError("Le modèle n'a pas encore été entraîné")
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance.head(top_n)
    
    def predict_score(self, user_features):
        """Prédit le score de crédit d'un utilisateur"""
        
        # Normalisation
        features_scaled = self.scaler.transform([user_features])
        
        # Prédiction
        proba = self.model.predict_proba(features_scaled)[0, 1]
        
        # Conversion en Neural Trust Score (0-100)
        neural_trust_score = proba * 100
        
        return neural_trust_score
    
    def save_model(self, model_dir='models/'):
        """Sauvegarde le modèle et les artefacts"""
        
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Modèle
        joblib.dump(self.model, f'{model_dir}neural_credit_model.pkl')
        
        # Scaler
        joblib.dump(self.scaler, f'{model_dir}scaler.pkl')
        
        # Encodeurs
        joblib.dump(self.label_encoders, f'{model_dir}label_encoders.pkl')
        
        # Métadonnées
        with open(f'{model_dir}model_metadata.json', 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        # Liste des features
        with open(f'{model_dir}feature_names.json', 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        print(f"\n💾 Modèle sauvegardé dans {model_dir}")
    
    def load_model(self, model_dir='models/'):
        """Charge un modèle sauvegardé"""
        
        self.model = joblib.load(f'{model_dir}neural_credit_model.pkl')
        self.scaler = joblib.load(f'{model_dir}scaler.pkl')
        self.label_encoders = joblib.load(f'{model_dir}label_encoders.pkl')
        
        with open(f'{model_dir}model_metadata.json', 'r') as f:
            self.model_metadata = json.load(f)
        
        with open(f'{model_dir}feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"✅ Modèle chargé depuis {model_dir}")

def main():
    """Pipeline complet d'entraînement"""
    
    print("=" * 60)
    print("🚀 NEURAL CREDIT - Entraînement du modèle de scoring")
    print("=" * 60)
    
    # Initialisation
    nc_model = NeuralCreditModel()
    
    # Chargement des données
    df = nc_model.load_data()
    
    # Préparation des features
    X, y = nc_model.prepare_features(df)
    
    # Entraînement
    model, auc = nc_model.train(X, y, use_xgboost=True)
    
    # Importance des features
    print("\n🎯 Top 15 features les plus importantes :")
    feature_importance = nc_model.get_feature_importance(15)
    print(feature_importance.to_string(index=False))
    
    # Sauvegarde
    nc_model.save_model()
    
    print("\n" + "=" * 60)
    print("✅ Pipeline d'entraînement terminé avec succès !")
    print("=" * 60)

if __name__ == "__main__":
    main()
