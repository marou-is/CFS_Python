import pandas as pd
import numpy as np
import os
import sys
import time

# Ajouter le chemin actuel pour importer les modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from DataLoader import DataLoader
from DataCleaner import DataCleaner
from SimilarityCalculator import SimilarityCalculator
from MovieRecommender import MovieRecommender

class ML100KRecommendationSystem:
    def __init__(self, data_path="dataset"):
        self.data_path = os.path.normpath(data_path)
        self.loader = None
        self.cleaner = None
        self.similarity_calc = None
        self.recommender = None

    def initialize_system(self):
        """Initialise le système avec les données MovieLens 100K"""
        print("🚀 INITIALISATION DU SYSTÈME MOVIELENS 100K")
        print("=" * 50)
        
        # 1. Chargement des données
        print("\n1. 📥 CHARGEMENT DES DONNÉES")
        self.loader = DataLoader(self.data_path)
        
        ratings, movies = self.loader.load_ml100k_data()
        
        if ratings is None:
            print("❌ Impossible de charger les données")
            return None
        
        # Afficher les informations des données
        self.loader.get_data_info()
        
        # 2. Nettoyage des données
        print("\n2. 🧹 NETTOYAGE DES DONNÉES")
        self.cleaner = DataCleaner(ratings, movies)
        self.cleaner.clean_ml100k_ratings()
        utility_matrix = self.cleaner.create_utility_matrix()
        
        # 3. Calcul des similarités
        print("\n3. 📐 CALCUL DES SIMILARITÉS")
        self.similarity_calc = SimilarityCalculator(utility_matrix)
        self.similarity_calc.calculate_user_similarity()
        self.similarity_calc.calculate_item_similarity()
        
        # 4. Initialisation du recommandeur
        print("\n4. 🤖 INITIALISATION DU RECOMMANDEUR")
        self.recommender = MovieRecommender(
            ratings=ratings,
            movies=movies,
            utility_matrix=utility_matrix,
            similarity_calculator=self.similarity_calc
        )
        
        print("✅ Système initialisé avec succès!")
        return self.recommender
    
    def get_available_users(self):
        """Retourne la liste des utilisateurs disponibles"""
        if self.recommender and hasattr(self.recommender, 'utility_matrix'):
            return sorted(self.recommender.utility_matrix.index.tolist())
        return []
    
    def get_recommendations(self, user_id=1, n_recommendations=5, method='hybrid'):
        """Génère des recommandations avec la méthode spécifiée"""
        print(f"\n🎯 RECOMMANDATIONS POUR L'UTILISATEUR {user_id} ({method.upper()})")
        print("=" * 60)
        
        if user_id not in self.recommender.utility_matrix.index:
            print(f"❌ L'utilisateur {user_id} n'existe pas dans la matrice utilitaire")
            available_users = self.get_available_users()[:10]
            print(f"   • Utilisateurs disponibles (10 premiers): {available_users}")
            return None
        
        # Afficher l'historique de l'utilisateur
        user_history = self.cleaner.get_user_rating_history(user_id, 5)
        if not user_history.empty:
            print(f"\n📖 HISTORIQUE RÉCENT DE L'UTILISATEUR {user_id}:")
            for _, rating in user_history.iterrows():
                title = rating.get('display_title', f"Movie {rating['movie_id']}")
                rating_date = rating['datetime'].strftime('%Y-%m-%d') if hasattr(rating['datetime'], 'strftime') else 'Unknown'
                print(f"   • {title}: {rating['rating']}★ ({rating_date})")
        
        # Générer les recommandations
        print(f"\n🔍 GÉNÉRATION DES RECOMMANDATIONS ({method})...")
        start_time = time.time()
        
        if method == 'user-based':
            recommendations = self.recommender.recommend_for_user(user_id, n_recommendations, 'user-based')
            method_name = "User-Based Collaborative Filtering"
        elif method == 'item-based':
            recommendations = self.recommender.recommend_for_user(user_id, n_recommendations, 'item-based')
            method_name = "Item-Based Collaborative Filtering"
        else:  # hybrid
            recommendations = self.recommender.hybrid_recommend(user_id, n_recommendations)
            method_name = "Hybrid (User-Based + Item-Based)"
        
        end_time = time.time()
        
        print(f"⏱️  Temps de génération: {end_time - start_time:.2f} secondes")
        print(f"🔧 Méthode utilisée: {method_name}")
        
        print(f"\n🌟 TOP {n_recommendations} RECOMMANDATIONS:")
        print("-" * 60)
        for i, (movie_id, title, score) in enumerate(recommendations, 1):
            movie_details = self.cleaner.get_movie_details(movie_id)
            clean_title = movie_details.get('clean_title', title)
            genres = movie_details.get('genres', [])
            
            print(f"{i:2d}. {clean_title}")
            print(f"    📊 Score: {score:.3f} | Genres: {', '.join(genres[:3])}")
            release_date = movie_details.get('release_date')
            if release_date and hasattr(release_date, 'year'):
                print(f"    🗓️  Année: {release_date.year}")
            print()
        
        # Sauvegarder les recommandations
        self.recommender.save_recommendations(user_id, recommendations, method)
        
        return recommendations

def show_most_active_users(system):
    """Affiche les utilisateurs les plus actifs"""
    print("\n🏆 UTILISATEURS LES PLUS ACTIFS")
    print("=" * 40)
    
    if system.recommender and hasattr(system.recommender, 'ratings'):
        user_activity = system.recommender.ratings['user_id'].value_counts().head(10)
        
        print("Top 10 des utilisateurs avec le plus de notes:")
        for i, (user_id, count) in enumerate(user_activity.items(), 1):
            user_ratings = system.recommender.ratings[system.recommender.ratings['user_id'] == user_id]
            avg_rating = user_ratings['rating'].mean()
            print(f"{i:2d}. Utilisateur {user_id}: {count:3d} notes | Moyenne: {avg_rating:.2f}★")

def performance_test(system):
    """Test de performance sur plusieurs utilisateurs et méthodes"""
    print("\n⚡ TEST DE PERFORMANCE COMPARATIVE")
    print("=" * 50)
    
    available_users = system.get_available_users()
    test_users = available_users[:3]  # Premier 3 utilisateurs
    methods = ['user-based', 'item-based', 'hybrid']
    
    print(f"Test sur {len(test_users)} utilisateurs avec {len(methods)} méthodes...")
    
    results = {}
    
    for method in methods:
        print(f"\n🔧 Méthode: {method.upper()}")
        print("-" * 30)
        
        total_time = 0
        method_results = []
        
        for user_id in test_users:
            print(f"   🧪 Utilisateur {user_id}...", end=" ")
            start_time = time.time()
            recommendations = system.get_recommendations(user_id, 3, method)
            end_time = time.time()
            user_time = end_time - start_time
            total_time += user_time
            
            if recommendations:
                avg_score = np.mean([rec[2] for rec in recommendations])
                method_results.append({
                    'user_id': user_id,
                    'time': user_time,
                    'avg_score': avg_score,
                    'n_recommendations': len(recommendations)
                })
                print(f"⏱️ {user_time:.2f}s | 📊 Score moyen: {avg_score:.3f}")
            else:
                print("❌ Échec")
        
        results[method] = {
            'total_time': total_time,
            'avg_time': total_time / len(test_users),
            'details': method_results
        }
    
    # Affichage des résultats comparatifs
    print(f"\n📊 RÉSULTATS PERFORMANCE COMPARATIVE:")
    print("=" * 50)
    print(f"{'Méthode':<15} {'Temps moyen':<12} {'Score moyen':<12} {'Efficacité'}")
    print("-" * 50)
    
    for method in methods:
        if method in results:
            avg_time = results[method]['avg_time']
            avg_scores = [r['avg_score'] for r in results[method]['details'] if 'avg_score' in r]
            overall_avg_score = np.mean(avg_scores) if avg_scores else 0
            efficiency = overall_avg_score / avg_time if avg_time > 0 else 0
            
            print(f"{method:<15} {avg_time:8.2f}s    {overall_avg_score:10.3f}    {efficiency:10.3f}")

def choose_method():
    """Demande à l'utilisateur de choisir une méthode"""
    print("\n🔧 CHOIX DE LA MÉTHODE DE RECOMMANDATION")
    print("=" * 40)
    print("1. 👥 User-Based Collaborative Filtering")
    print("   → Basé sur la similarité entre utilisateurs")
    print("2. 🎬 Item-Based Collaborative Filtering") 
    print("   → Basé sur la similarité entre films")
    print("3. 🌟 Hybrid (User-Based + Item-Based)")
    print("   → Combinaison des deux méthodes (recommandé)")
    print("4. 🔄 Comparaison des 3 méthodes")
    
    while True:
        choice = input("\nChoisissez la méthode (1/2/3/4): ").strip()
        
        if choice == "1":
            return 'user-based'
        elif choice == "2":
            return 'item-based'
        elif choice == "3":
            return 'hybrid'
        elif choice == "4":
            return 'compare'
        else:
            print("❌ Choix invalide. Veuillez choisir 1, 2, 3 ou 4.")

def compare_methods(system, user_id):
    """Compare les 3 méthodes pour un utilisateur"""
    print(f"\n🔄 COMPARAISON DES MÉTHODES POUR L'UTILISATEUR {user_id}")
    print("=" * 60)
    
    methods = ['user-based', 'item-based', 'hybrid']
    all_recommendations = {}
    
    for method in methods:
        print(f"\n🔧 {method.upper()}:")

        recommendations = system.get_recommendations(user_id, 3, method)
        if recommendations:
            all_recommendations[method] = recommendations
            
            # Afficher les top 3
            for i, (movie_id, title, score) in enumerate(recommendations, 1):
                movie_details = system.cleaner.get_movie_details(movie_id)
                clean_title = movie_details.get('clean_title', title)
                print(f"   {i}. {clean_title} (score: {score:.3f})")
    
    # Analyse comparative
    print(f"\n📊 ANALYSE COMPARATIVE:")
    print("-" * 40)
    
    common_movies = {}
    for method, recs in all_recommendations.items():
        movie_ids = [rec[0] for rec in recs]
        for movie_id in movie_ids:
            if movie_id not in common_movies:
                common_movies[movie_id] = []
            common_movies[movie_id].append(method)
    
    # Films recommandés par plusieurs méthodes
    print("🎯 Films recommandés par plusieurs méthodes:")
    for movie_id, methods_list in common_movies.items():
        if len(methods_list) > 1:
            movie_details = system.cleaner.get_movie_details(movie_id)
            clean_title = movie_details.get('clean_title', f"Movie {movie_id}")
            print(f"   • {clean_title} → {', '.join(methods_list)}")

def interactive_mode(system):
    """Mode interactif où l'utilisateur choisit l'ID et la méthode"""
    print("\n🔍 MODE INTERACTIF")
    print("=" * 30)
    
    available_users = system.get_available_users()
    print(f"👥 Utilisateurs disponibles: {len(available_users)} (IDs: {available_users[0]} à {available_users[-1]})")
    
    while True:
        print("\nOptions:")
        print("1. 🎯 Recommandations pour un utilisateur")
        print("2. ⚡ Test de performance comparatif")
        print("3. 🏆 Utilisateurs les plus actifs")
        print("4. ❌ Quitter")
        
        choice = input("\nVotre choix (1/2/3/4): ").strip()
        
        if choice == "1":
            try:
                user_id = int(input("🎯 Entrez l'ID utilisateur: ").strip())
                if user_id in available_users:
                    method_choice = choose_method()
                    
                    if method_choice == 'compare':
                        compare_methods(system, user_id)
                    else:
                        system.get_recommendations(user_id, n_recommendations=5, method=method_choice)
                else:
                    print(f"❌ Utilisateur {user_id} non trouvé.")
                    print(f"   IDs valides: {available_users[0]} à {available_users[-1]}")
            except ValueError:
                print("❌ Veuillez entrer un nombre valide.")
        
        elif choice == "2":
            performance_test(system)
        
        elif choice == "3":
            show_most_active_users(system)
        
        elif choice == "4":
            print("👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide.")

def diagnostic():
    """Fonction de diagnostic du système"""
    print("🔍 DIAGNOSTIC DU SYSTÈME")
    print("=" * 40)
    
    print(f"1. 📁 Dossier de travail: {os.getcwd()}")
    print(f"2. 📁 Contenu du dossier:")
    
    try:
        items = os.listdir('.')
        for item in items:
            full_path = os.path.join('.', item)
            if os.path.isdir(full_path):
                print(f"   📁 {item}/")
                if item == "dataset":
                    print(f"     Contenu de dataset/:")
                    try:
                        for sub_item in os.listdir(full_path):
                            print(f"       - {sub_item}")
                    except Exception as e:
                        print(f"       ❌ Erreur: {e}")
            else:
                print(f"   📄 {item}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    print(f"3. 🐍 Version Python: {sys.version}")
    print(f"4. 📦 Modules importés:")
    try:
        import pandas as pd
        print(f"   ✅ pandas: {pd.__version__}")
    except:
        print(f"   ❌ pandas: NON INSTALLÉ")
    
    try:
        import numpy as np
        print(f"   ✅ numpy: {np.__version__}")
    except:
        print(f"   ❌ numpy: NON INSTALLÉ")
    
    try:
        import sklearn
        print(f"   ✅ scikit-learn: {sklearn.__version__}")
    except:
        print(f"   ❌ scikit-learn: NON INSTALLÉ")

def main():
    """Fonction principale"""
    print("🎬 SYSTÈME DE RECOMMANDATION MOVIELENS")
    print("=" * 50)
    
    # Vérifier que le dossier de données existe
    data_path = "dataset"
    data_path = os.path.normpath(data_path)
    
    print(f"📁 Recherche du dossier de données: {data_path}")
    print(f"📁 Dossier de travail actuel: {os.getcwd()}")
    
    if not os.path.exists(data_path):
        print(f"❌ Dossier de données '{data_path}' non trouvé!")
        print(f"\n📥 SOLUTIONS:")
        print(f"1. Vérifiez que le dossier 'dataset' existe")
        print(f"2. Fichiers attendus dans dataset/:")
        print(f"   • u.data")
        print(f"   • u.item")
        print(f"   • u.user")
        
        print(f"\n📂 Contenu du dossier actuel:")
        try:
            for item in os.listdir('.'):
                item_path = os.path.join('.', item)
                if os.path.isdir(item_path):
                    print(f"   📁 {item}/")
                else:
                    print(f"   📄 {item}")
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
        
        return
    
    # Initialiser le système
    system = ML100KRecommendationSystem(data_path)
    recommender = system.initialize_system()
    
    if recommender is None:
        print("❌ Échec de l'initialisation du système")
        return
    
    # Mode interactif
    interactive_mode(system)

if __name__ == "__main__":
    # Menu principal simplifié
    print("🔧 SYSTÈME DE RECOMMANDATION INTERACTIF")
    print("Choisissez le mode:")
    print("1 = 🎯 Mode interactif (CHOISIR UTILISATEUR & MÉTHODE)")
    print("2 = 🔍 Diagnostic système")
    
    choix = input("Mode (1/2): ").strip()
    
    if choix == "2":
        diagnostic()
    else:
        main()