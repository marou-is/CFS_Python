import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class MovieRecommender:
    def __init__(self, ratings, movies, utility_matrix, similarity_calculator):
        self.ratings = ratings
        self.movies = movies
        self.utility_matrix = utility_matrix
        self.similarity_calc = similarity_calculator
        self.predictions = {}
    def user_based_predict(self, user_id, movie_id, k=20):
        """Prédit une note avec la méthode user-based collaborative filtering"""
        # Vérifier si l'utilisateur ou le film existe
        if user_id not in self.utility_matrix.index or movie_id not in self.utility_matrix.columns:
            return self._get_default_rating(user_id)
        
        # Notes de l'utilisateur cible
        user_ratings = self.utility_matrix.loc[user_id]
        
        # Utilisateurs qui ont noté ce film
        if movie_id not in self.utility_matrix.columns:
            return self._get_user_average_rating(user_id)
            
        movie_ratings = self.utility_matrix[movie_id]
        users_rated = movie_ratings[movie_ratings > 0].index
        
        if len(users_rated) == 0:
            return self._get_user_average_rating(user_id)
        
        # Calculer les similarités avec les utilisateurs qui ont noté le film
        similarities = []
        ratings_list = []
        
        for other_user in users_rated:
            if other_user != user_id:
                try:
                    sim = self.similarity_calc.user_similarity_df.loc[user_id, other_user]
                    # Ne considérer que les similarités positives
                    if sim > 0:
                        similarities.append(sim)
                        ratings_list.append(movie_ratings[other_user])
                except KeyError:
                    continue
        
        if not similarities or len(similarities) < 2:
            return self._get_user_average_rating(user_id)
        
        # Prendre les k plus proches voisins
        if len(similarities) > k:
            indices = np.argsort(similarities)[::-1][:k]
            similarities = [similarities[i] for i in indices]
            ratings_list = [ratings_list[i] for i in indices]
        
        # Éviter la division par zéro
        if sum(similarities) == 0:
            return self._get_user_average_rating(user_id)
        
        # Prédiction pondérée par similarité
        weighted_sum = np.dot(similarities, ratings_list)
        prediction = weighted_sum / sum(similarities)
        
        return np.clip(prediction, 1, 5)
    
    def item_based_predict(self, user_id, movie_id, k=20):
        """Prédit une note avec la méthode item-based collaborative filtering"""
        if user_id not in self.utility_matrix.index or movie_id not in self.utility_matrix.columns:
            return 3.0
        
        # Notes de l'utilisateur
        user_ratings = self.utility_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        
        if len(rated_movies) == 0:
            return 3.0
        
        # Calculer les similarités avec les films notés par l'utilisateur
        similarities = []
        ratings_list = []

        for other_movie in rated_movies:
            if other_movie != movie_id:
                try:
                    sim = self.similarity_calc.item_similarity_df.loc[movie_id, other_movie]
                    if sim > 0:
                        similarities.append(sim)
                        ratings_list.append(user_ratings[other_movie])
                except KeyError:
                    continue
        
        if not similarities or len(similarities) < 2:
            return self._get_user_average_rating(user_id)
        
        # Prendre les k plus proches voisins
        if len(similarities) > k:
            indices = np.argsort(similarities)[::-1][:k]
            similarities = [similarities[i] for i in indices]
            ratings_list = [ratings_list[i] for i in indices]
        
        if sum(similarities) == 0:
            return np.mean(ratings_list) if ratings_list else 3.0
        
        # Prédiction pondérée
        weighted_sum = np.dot(similarities, ratings_list)
        prediction = weighted_sum / sum(similarities)
        
        return np.clip(prediction, 1, 5)
    
    def _get_user_average_rating(self, user_id):
        """Retourne la note moyenne d'un utilisateur"""
        if user_id in self.utility_matrix.index:
            user_ratings = self.utility_matrix.loc[user_id]
            rated_movies = user_ratings[user_ratings > 0]
            return rated_movies.mean() if len(rated_movies) > 0 else 3.0
        return 3.0
    
    def _get_default_rating(self, user_id):
        """Retourne une note par défaut"""
        return self._get_user_average_rating(user_id)
    
    def recommend_for_user(self, user_id, n_recommendations=10, method='user-based'):
        """Génère des recommandations pour un utilisateur"""
        # Films déjà notés par l'utilisateur
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        rated_movies = user_ratings['movie_id'].tolist()
        
        # Films disponibles dans la matrice utilitaire
        available_movies = self.utility_matrix.columns.tolist()
        
        # Films non notés
        unrated_movies = [movie for movie in available_movies if movie not in rated_movies]
        
        if not unrated_movies:
            return []
        
        print(f"🔍 Génération de recommandations pour l'utilisateur {user_id}...")
        print(f"   • Films notés: {len(rated_movies)}")
        print(f"   • Films à évaluer: {len(unrated_movies)}")
        
        # Prédire les notes pour les films non notés (limiter pour la performance)
        predictions = []
        max_movies_to_predict = min(500, len(unrated_movies))
        
        for i, movie_id in enumerate(unrated_movies[:max_movies_to_predict]):
            if method == 'user-based':
                predicted_rating = self.user_based_predict(user_id, movie_id)
            else:
                predicted_rating = self.item_based_predict(user_id, movie_id)
            
            # Trouver le titre du film
            movie_title = f"Movie {movie_id}"
            if self.movies is not None:
                movie_data = self.movies[self.movies['movie_id'] == movie_id]
                if len(movie_data) > 0:
                    movie_title = movie_data.iloc[0].get('title', f"Movie {movie_id}")
            
            predictions.append((movie_id, movie_title, predicted_rating))
            
            # Barre de progression
            if (i + 1) % 100 == 0:
                print(f"   • Progression: {i + 1}/{max_movies_to_predict}")
        
        # Trier par note prédite décroissante
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        return predictions[:n_recommendations]
    
    def hybrid_recommend(self, user_id, n_recommendations=10, user_weight=0.6):
        """Recommandation hybride combinant user-based et item-based"""
        print(f"🌟 Génération de recommandations hybrides...")
        
        # Obtenir les recommandations des deux méthodes
        user_recs = self.recommend_for_user(user_id, n_recommendations * 3, 'user-based')
        item_recs = self.recommend_for_user(user_id, n_recommendations * 3, 'item-based')
        
        # Combiner les scores
        combined_scores = {}
        
        for movie_id, title, score in user_recs:
            combined_scores[(movie_id, title)] = combined_scores.get((movie_id, title), 0) + score * user_weight
        
        for movie_id, title, score in item_recs:
            combined_scores[(movie_id, title)] = combined_scores.get((movie_id, title), 0) + score * (1 - user_weight)
        
        # Convertir en liste et trier
        hybrid_recs = [(movie_id, title, score) for (movie_id, title), score in combined_scores.items()]
        hybrid_recs.sort(key=lambda x: x[2], reverse=True)
        
        return hybrid_recs[:n_recommendations]
    
    def evaluate_model(self, test_size=0.2, method='user-based', sample_size=1000):
        """Évalue la performance du modèle"""
        print(f"\n🧪 ÉVALUATION DU MODÈLE ({method.upper()})")
        print("=" * 40)
        
        # Séparation train-test
        train_df, test_df = train_test_split(self.ratings, test_size=test_size, random_state=42)
        
        # Réentraîner sur les données d'entraînement
        from data_cleaning import DataCleaner
        from similarity import SimilarityCalculator
        
        train_cleaner = DataCleaner(train_df, self.movies)
        train_cleaner.clean_ratings_data()
        train_matrix = train_cleaner.create_utility_matrix()
        
        train_similarity = SimilarityCalculator(train_matrix)
        train_similarity.calculate_user_similarity()
        train_similarity.calculate_item_similarity()
        
        train_recommender = MovieRecommender(train_df, self.movies, train_matrix, train_similarity)
        
        # Échantillonner le test set pour accélérer
        test_sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)
        
        predictions = []
        actual_ratings = []
        
        print("📊 Calcul des prédictions sur l'ensemble de test...")
        for i, (_, row) in enumerate(test_sample.iterrows()):
            user_id = row['user_id']
            movie_id = row['movie_id']
            actual_rating = row['rating']
            
            if method == 'user-based':
                predicted_rating = train_recommender.user_based_predict(user_id, movie_id)
            else:
                predicted_rating = train_recommender.item_based_predict(user_id, movie_id)
            
            predictions.append(predicted_rating)
            actual_ratings.append(actual_rating)
            
            if (i + 1) % 200 == 0:
                print(f"   • {i + 1}/{len(test_sample)} prédictions calculées")
        
        # Métriques d'évaluation
        mae = mean_absolute_error(actual_ratings, predictions)
        rmse = np.sqrt(mean_squared_error(actual_ratings, predictions))
        
        # Précision@k
        precision_at_5 = self._calculate_precision_at_k(actual_ratings, predictions, k=5)
        precision_at_10 = self._calculate_precision_at_k(actual_ratings, predictions, k=10)
        
        print(f"\n📈 RÉSULTATS DE L'ÉVALUATION:")
        print(f"   • MAE:  {mae:.3f}")
        print(f"   • RMSE: {rmse:.3f}")
        print(f"   • Precision@5:  {precision_at_5:.3f}")
        print(f"   • Precision@10: {precision_at_10:.3f}")
        print(f"   • Taille échantillon: {len(test_sample)}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'precision@5': precision_at_5,
            'precision@10': precision_at_10,
            'predictions': predictions,
            'actual': actual_ratings
        }
    
    def _calculate_precision_at_k(self, actual, predicted, k=10, threshold=4.0):
        """Calcule la précision@k"""
        if len(predicted) == 0:
            return 0.0
            
        # Trier les prédictions et prendre les top-k
        indices = np.argsort(predicted)[::-1][:k]
        top_k_actual = [actual[i] for i in indices]
        
        # Compter combien ont une note réelle >= threshold
        relevant_count = sum(1 for rating in top_k_actual if rating >= threshold)
        
        return relevant_count / k if k > 0 else 0.0
    
    def save_recommendations(self, user_id, recommendations, method, path="./results"):
        """Sauvegarde les recommandations dans un fichier"""
        import os
        import datetime
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{path}/recommendations_user_{user_id}_{method}_{timestamp}.csv"
        
        # Créer un DataFrame avec les recommandations
        rec_df = pd.DataFrame(recommendations, columns=['movie_id', 'title', 'predicted_rating'])
        rec_df['user_id'] = user_id
        rec_df['method'] = method
        rec_df['timestamp'] = timestamp
        
        # Sauvegarder
        rec_df.to_csv(filename, index=False)
        print(f"✅ Recommandations sauvegardées: {filename}")
        
        return filename