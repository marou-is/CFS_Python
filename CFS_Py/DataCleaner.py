import pandas as pd
import numpy as np
from datetime import datetime

class DataCleaner:
    def __init__(self, ratings, movies):
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        self.utility_matrix = None
        
    def clean_ml100k_ratings(self):
        """Nettoie spécifiquement les données MovieLens 100K"""
        print("🧹 Nettoyage des données MovieLens 100K...")
        
        # Vérifier les valeurs manquantes
        missing_before = self.ratings.isnull().sum().sum()
        self.ratings = self.ratings.dropna()
        missing_after = self.ratings.isnull().sum().sum()
        print(f"   • Valeurs manquantes supprimées: {missing_before - missing_after}")
        
        # S'assurer que les types sont corrects
        self.ratings['user_id'] = self.ratings['user_id'].astype(int)
        self.ratings['movie_id'] = self.ratings['movie_id'].astype(int)
        self.ratings['rating'] = self.ratings['rating'].astype(float)
        
        # Convertir le timestamp en datetime
        self.ratings['datetime'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        
        # Nettoyer les données de films si disponibles
        if self.movies is not None:
            self._clean_movie_data()
        
        print(f"✅ Données MovieLens 100K nettoyées: {self.ratings.shape}")
        return self.ratings
    
    def clean_ratings_data(self):
        """Alias pour la compatibilité"""
        return self.clean_ml100k_ratings()
        
    def _clean_movie_data(self):
        """Nettoie les données de films"""
        # Nettoyer les titres - enlever l'année entre parenthèses pour l'affichage
        if 'title' in self.movies.columns:
            self.movies['clean_title'] = self.movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)
        
        # Convertir les dates de sortie
        if 'release_date' in self.movies.columns:
            self.movies['release_date'] = pd.to_datetime(self.movies['release_date'], errors='coerce')
    
    def create_utility_matrix(self, min_ratings_per_user=5, min_ratings_per_movie=10):
        """Crée la matrice utilitaire user-item pour MovieLens"""
        print("📊 Création de la matrice utilitaire MovieLens...")
        
        # Filtrer les utilisateurs et films avec trop peu de notations
        user_rating_counts = self.ratings['user_id'].value_counts()
        movie_rating_counts = self.ratings['movie_id'].value_counts()
        
        filtered_users = user_rating_counts[user_rating_counts >= min_ratings_per_user].index
        filtered_movies = movie_rating_counts[movie_rating_counts >= min_ratings_per_movie].index
        
        filtered_ratings = self.ratings[
            (self.ratings['user_id'].isin(filtered_users)) & 
            (self.ratings['movie_id'].isin(filtered_movies))
        ]
        
        print(f"   • Utilisateurs après filtrage: {len(filtered_users)}/{self.ratings['user_id'].nunique()}")
        print(f"   • Films après filtrage: {len(filtered_movies)}/{self.ratings['movie_id'].nunique()}")
        print(f"   • Notations après filtrage: {len(filtered_ratings)}/{len(self.ratings)}")
        
        # Créer la matrice utilitaire
        self.utility_matrix = filtered_ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0  # 0 pour les valeurs manquantes
        )
        
        # Calculer la densité
        total_cells = self.utility_matrix.shape[0] * self.utility_matrix.shape[1]
        rated_cells = (self.utility_matrix > 0).sum().sum()
        density = (rated_cells / total_cells) * 100
        
        print(f"✅ Matrice utilitaire MovieLens créée: {self.utility_matrix.shape}")
        print(f"   • Densité: {density:.4f}%")
        print(f"   • Notes moyennes par utilisateur: {rated_cells / self.utility_matrix.shape[0]:.1f}")
        print(f"   • Notes moyennes par film: {rated_cells / self.utility_matrix.shape[1]:.1f}")
        
        return self.utility_matrix
    
    def get_user_rating_history(self, user_id, n=10):
        """Retourne l'historique des notations d'un utilisateur"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id].sort_values('datetime', ascending=False)
        
        if self.movies is not None and 'clean_title' in self.movies.columns:
            user_ratings = user_ratings.merge(self.movies[['movie_id', 'clean_title']], on='movie_id', how='left')
            user_ratings['display_title'] = user_ratings['clean_title'].fillna(user_ratings['movie_id'].astype(str))
        else:
            user_ratings['display_title'] = 'Movie ' + user_ratings['movie_id'].astype(str)
        
        return user_ratings.head(n)
    
    def get_movie_details(self, movie_id):
        """Retourne les détails complets d'un film"""
        if self.movies is None:
            return {"title": f"Movie {movie_id}", "movie_id": movie_id}
        
        movie_row = self.movies[self.movies['movie_id'] == movie_id]
        if len(movie_row) == 0:
            return {"title": f"Movie {movie_id}", "movie_id": movie_id}
            
        movie_info = movie_row.iloc[0]
        
        # Extraire les genres
        genres = []
        if 'Action' in self.movies.columns:
            genre_columns = self.movies.columns[6:24]
            for genre in genre_columns:
                if genre in movie_info and movie_info[genre] == 1:
                    genres.append(genre)
        
        details = {
            'movie_id': movie_id,
            'title': movie_info.get('title', f'Movie {movie_id}'),
            'clean_title': movie_info.get('clean_title', f'Movie {movie_id}'),
            'release_date': movie_info.get('release_date', 'Unknown'),
            'genres': genres,
            'IMDb_URL': movie_info.get('IMDb_URL', '')
        }
        
        return details