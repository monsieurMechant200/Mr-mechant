"""
Neural Credit - Générateur de données simulées pour le Cameroun
Auteur: David Meilleur
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Configuration
np.random.seed(42)
random.seed(42)

# Paramètres réalistes pour le Cameroun
N_USERS = 5000
REGIONS = ['Douala', 'Yaoundé', 'Bafoussam', 'Garoua', 'Bamenda', 'Maroua']
MOBILE_OPERATORS = ['Orange Money', 'MTN MoMo', 'Express Union']
PROFESSIONS = ['Commerçant', 'Étudiant', 'Fonctionnaire', 'Artisan', 'Chauffeur', 'Enseignant', 'Informel']

def generate_user_profile(user_id):
    """Génère le profil d'un utilisateur"""
    age = np.random.randint(18, 65)
    gender = np.random.choice(['M', 'F'])
    region = np.random.choice(REGIONS)
    profession = np.random.choice(PROFESSIONS)
    
    # Revenu mensuel approximatif (en FCFA)
    if profession == 'Fonctionnaire':
        monthly_income = np.random.randint(150000, 500000)
    elif profession == 'Étudiant':
        monthly_income = np.random.randint(30000, 100000)
    elif profession == 'Commerçant':
        monthly_income = np.random.randint(100000, 800000)
    else:
        monthly_income = np.random.randint(50000, 300000)
    
    return {
        'user_id': f'USER_{user_id:05d}',
        'age': age,
        'gender': gender,
        'region': region,
        'profession': profession,
        'monthly_income': monthly_income,
        'registration_date': datetime.now() - timedelta(days=np.random.randint(30, 365))
    }

def generate_mobile_money_transactions(user_id, profile, n_months=6):
    """Génère l'historique Mobile Money"""
    transactions = []
    income = profile['monthly_income']
    
    for month in range(n_months):
        # Nombre de transactions par mois (proportionnel au revenu)
        n_tx = int(np.random.poisson(lam=max(5, income/50000)))
        
        for _ in range(n_tx):
            tx_type = np.random.choice(['Dépôt', 'Retrait', 'Transfert', 'Paiement'], 
                                       p=[0.3, 0.4, 0.2, 0.1])
            
            # Montant variable selon le type
            if tx_type == 'Dépôt':
                amount = np.random.uniform(5000, income * 0.5)
            elif tx_type == 'Retrait':
                amount = np.random.uniform(2000, income * 0.3)
            elif tx_type == 'Transfert':
                amount = np.random.uniform(1000, income * 0.2)
            else:  # Paiement
                amount = np.random.uniform(500, 50000)
            
            transactions.append({
                'user_id': user_id,
                'tx_type': tx_type,
                'amount': round(amount, 2),
                'operator': np.random.choice(MOBILE_OPERATORS),
                'timestamp': profile['registration_date'] + timedelta(days=month*30 + np.random.randint(0, 30)),
                'hour': np.random.randint(6, 23)
            })
    
    return transactions

def generate_behavioral_features(user_id, profile):
    """Génère les features comportementales"""
    
    # Stabilité géographique (0-1)
    geo_stability = np.random.beta(5, 2)  # Tendance vers la stabilité
    
    # Activité téléphonique hebdomadaire
    calls_per_week = int(np.random.poisson(lam=20))
    
    # Régularité des habitudes (0-1)
    habit_consistency = np.random.beta(4, 2)
    
    # Score psychométrique simulé (réponses à 10 questions, 1-5)
    psy_responses = [np.random.randint(1, 6) for _ in range(10)]
    psy_score = np.mean(psy_responses) / 5.0  # Normalisation
    
    # Engagement social (posts par semaine)
    social_activity = int(np.random.poisson(lam=5))
    
    return {
        'user_id': user_id,
        'geo_stability': round(geo_stability, 3),
        'calls_per_week': calls_per_week,
        'habit_consistency': round(habit_consistency, 3),
        'psy_score': round(psy_score, 3),
        'social_activity': social_activity
    }

def generate_credit_label(profile, transactions_df, behavioral):
    """Génère le label de crédit (bon/mauvais payeur)"""
    
    # Calcul de features agrégées
    avg_balance = transactions_df['amount'].mean()
    tx_frequency = len(transactions_df) / 6  # par mois
    tx_variance = transactions_df['amount'].std()
    
    # Score composite (logique métier simplifiée)
    score = 0
    
    # Revenu
    if profile['monthly_income'] > 200000:
        score += 0.3
    elif profile['monthly_income'] > 100000:
        score += 0.2
    else:
        score += 0.1
    
    # Stabilité des transactions
    if tx_variance < avg_balance * 0.5:
        score += 0.2
    
    # Fréquence raisonnable
    if 5 < tx_frequency < 50:
        score += 0.2
    
    # Comportement
    score += behavioral['geo_stability'] * 0.15
    score += behavioral['habit_consistency'] * 0.1
    score += behavioral['psy_score'] * 0.05
    
    # Ajout de bruit réaliste
    score += np.random.normal(0, 0.1)
    score = np.clip(score, 0, 1)
    
    # Label binaire
    is_good_payer = 1 if score > 0.55 else 0
    
    return {
        'user_id': profile['user_id'],
        'credit_score': round(score, 3),
        'is_good_payer': is_good_payer
    }

def main():
    """Génération complète du dataset"""
    
    print("🚀 Génération du dataset Neural Credit...")
    
    # 1. Profils utilisateurs
    users = [generate_user_profile(i) for i in range(N_USERS)]
    users_df = pd.DataFrame(users)
    
    # 2. Transactions Mobile Money
    all_transactions = []
    for user in users:
        txs = generate_mobile_money_transactions(user['user_id'], user)
        all_transactions.extend(txs)
    
    transactions_df = pd.DataFrame(all_transactions)
    
    # 3. Features comportementales
    behavioral = [generate_behavioral_features(user['user_id'], user) for user in users]
    behavioral_df = pd.DataFrame(behavioral)
    
    # 4. Labels de crédit
    labels = []
    for user in users:
        user_tx = transactions_df[transactions_df['user_id'] == user['user_id']]
        user_behavior = behavioral_df[behavioral_df['user_id'] == user['user_id']].iloc[0].to_dict()
        label = generate_credit_label(user, user_tx, user_behavior)
        labels.append(label)
    
    labels_df = pd.DataFrame(labels)
    
    # 5. Fusion des données
    final_df = users_df.merge(behavioral_df, on='user_id').merge(labels_df, on='user_id')
    
    # 6. Sauvegarde
    final_df.to_csv('data/users_features.csv', index=False)
    transactions_df.to_csv('data/transactions.csv', index=False)
    
    print(f"✅ Dataset généré avec succès !")
    print(f"   - {len(users_df)} utilisateurs")
    print(f"   - {len(transactions_df)} transactions")
    print(f"   - Taux de bons payeurs : {labels_df['is_good_payer'].mean()*100:.1f}%")
    
    # Statistiques descriptives
    print("\n📊 Statistiques du dataset :")
    print(final_df.describe())
    
    return final_df

if __name__ == "__main__":
    main()
