import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

class SimilarityCalculator:
    def __init__(self, utility_matrix):
        self.utility_matrix = utility_matrix
        self.user_similarity = None
        self.item_similarity = None
        self.user_similarity_df = None
        self.item_similarity_df = None
        
    def calculate_user_similarity(self, method='cosine'):
        """Calcule la similarité entre utilisateurs"""
        print("👥 Calcul de la similarité entre utilisateurs...")
        
        if method == 'cosine':
            # Utiliser une matrice sparse pour optimiser la mémoire
            sparse_matrix = csr_matrix(self.utility_matrix.values)
            self.user_similarity = cosine_similarity(sparse_matrix)
            
            # Créer un DataFrame avec les IDs utilisateurs
            self.user_similarity_df = pd.DataFrame(
                self.user_similarity,
                index=self.utility_matrix.index,
                columns=self.utility_matrix.index
            )
        
        print(f"✅ Similarité utilisateur calculée ({method})")
        print(f"   • Dimensions: {self.user_similarity_df.shape}")
        if self.user_similarity_df.values.size > 0:
            print(f"   • Plage des similarités: {self.user_similarity_df.values.min():.3f} - {self.user_similarity_df.values.max():.3f}")
        
        return self.user_similarity_df
    
    def calculate_item_similarity(self, method='cosine'):
        """Calcule la similarité entre films"""
        print("🎬 Calcul de la similarité entre films...")
        
        # Transposer la matrice pour avoir les films en lignes
        item_matrix = self.utility_matrix.T
        
        if method == 'cosine':
            item_sparse = csr_matrix(item_matrix.values)
            self.item_similarity = cosine_similarity(item_sparse)
            
            self.item_similarity_df = pd.DataFrame(
                self.item_similarity,
                index=item_matrix.index,
                columns=item_matrix.index
            )
        
        print(f"✅ Similarité film calculée ({method})")
        print(f"   • Dimensions: {self.item_similarity_df.shape}")
        if self.item_similarity_df.values.size > 0:
            print(f"   • Plage des similarités: {self.item_similarity_df.values.min():.3f} - {self.item_similarity_df.values.max():.3f}")
        
        return self.item_similarity_df
    
    def get_most_similar_users(self, user_id, n=5):
        """Retourne les utilisateurs les plus similaires"""
        if self.user_similarity_df is None:
            self.calculate_user_similarity()
        
        if user_id not in self.user_similarity_df.index:
            print(f"❌ Utilisateur {user_id} non trouvé")
            return pd.Series()
        
        # Obtenir les similarités et exclure l'utilisateur lui-même
        user_similarities = self.user_similarity_df.loc[user_id].drop(user_id, errors='ignore')
        most_similar = user_similarities.sort_values(ascending=False).head(n)
        
        return most_similar
    
    def get_most_similar_items(self, movie_id, n=5):
        """Retourne les films les plus similaires"""
        if self.item_similarity_df is None:
            self.calculate_item_similarity()
        
        if movie_id not in self.item_similarity_df.index:
            print(f"❌ Film {movie_id} non trouvé")
            return pd.Series()
        
        # Obtenir les similarités et exclure le film lui-même
        item_similarities = self.item_similarity_df.loc[movie_id].drop(movie_id, errors='ignore')
        most_similar = item_similarities.sort_values(ascending=False).head(n)
        
        return most_similar
    
    def reduce_dimensionality(self, n_components=50, method='svd'):
        """Réduit la dimensionalité pour améliorer les calculs"""
        print(f"🔽 Réduction de dimensionalité ({method}, {n_components} composantes)...")
        
        if method == 'svd':
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_matrix = svd.fit_transform(self.utility_matrix.values)
            
            print(f"✅ Réduction SVD terminée")
            print(f"   • Variance expliquée: {svd.explained_variance_ratio_.sum():.3f}")
            
            return reduced_matrix, svd
    
    def save_similarity_matrices(self, path="./results"):
        """Sauvegarde les matrices de similarité"""
        import os
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.user_similarity_df is not None:
            # Sauvegarder un échantillon de la matrice utilisateur
            user_sample = self.user_similarity_df.iloc[:50, :50]
            user_sample.to_csv(f"{path}/user_similarity_sample.csv")
            print(f"✅ Matrice de similarité utilisateur échantillon sauvegardée")
        
        if self.item_similarity_df is not None:
            # Sauvegarder un échantillon de la matrice item
            item_sample = self.item_similarity_df.iloc[:50, :50]
            item_sample.to_csv(f"{path}/item_similarity_sample.csv")
            print(f"✅ Matrice de similarité film échantillon sauvegardée")