import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path="dataset"):
        self.data_path = os.path.normpath(data_path)
        self.ratings = None
        self.movies = None
    
    def load_ml100k_data(self):
        """Charge les données MovieLens 100K"""
        print("🎬 Chargement des données MovieLens 100K...")
        
        # Charger les notations (u.data)
        ratings_file = os.path.join(self.data_path, "u.data")
        if os.path.exists(ratings_file):
            self.ratings = pd.read_csv(ratings_file, sep='\t', 
                                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
        else:
            print(f"❌ Fichier {ratings_file} non trouvé")
            return None, None
        
        # Charger les informations sur les films (u.item)
        movies_file = os.path.join(self.data_path, "u.item")
        if os.path.exists(movies_file):
            self.movies = pd.read_csv(movies_file, sep='|', encoding='latin-1',
                                    names=['movie_id', 'title', 'release_date', 'video_release_date',
                                           'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                           'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                                           'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                                           'Thriller', 'War', 'Western'])

        print(f"✅ Données MovieLens 100K chargées:")
        print(f"   • Ratings: {self.ratings.shape}")
        print(f"   • Movies: {self.movies.shape}")
        
        return self.ratings, self.movies
    
    def load_all_data(self):
        """Charge toutes les données"""
        ratings, movies = self.load_ml100k_data()
        users = self.load_user_info()
        return ratings, movies, users
    
    def load_specific_split(self, split_name="ua"):
        """Charge un split spécifique (ua, ub, u1, u2, etc.)"""
        base_file = os.path.join(self.data_path, f"{split_name}.base")
        test_file = os.path.join(self.data_path, f"{split_name}.test")
        
        if not os.path.exists(base_file) or not os.path.exists(test_file):
            print(f" Fichiers de split {split_name} non trouvés")
            return None, None
        
        # Charger les données d'entraînement
        train_ratings = pd.read_csv(base_file, sep='\t', 
                                  names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        # Charger les données de test
        test_ratings = pd.read_csv(test_file, sep='\t',
                                 names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        print(f"✅ Split {split_name} chargé:")
        print(f"   • Train: {train_ratings.shape}")
        print(f"   • Test: {test_ratings.shape}")
        
        return train_ratings, test_ratings
  
    def load_user_info(self):
        """Charge les informations utilisateurs (u.user)"""
        user_file = os.path.join(self.data_path, "u.user")
        if os.path.exists(user_file):
            users = pd.read_csv(user_file, sep='|', 
                              names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
            print(f"✅ Informations utilisateurs chargées: {users.shape}")
            return users
        else:
            print(" Fichier u.user non trouvé")
            return None
    
    def load_genre_info(self):
        """Charge les informations sur les genres (u.genre)"""
        genre_file = os.path.join(self.data_path, "u.genre")
        if os.path.exists(genre_file):
            genres = []
            with open(genre_file, 'r', encoding='latin-1') as f:
                for line in f:
                    if line.strip():
                        genre, idx = line.strip().split('|')
                        genres.append(genre)
            print(f" Genres chargés: {len(genres)} genres")
            return genres
        else:
            print(" Fichier u.genre non trouvé")
            return None
    
    def get_data_info(self):
        """Affiche des informations détaillées sur les données"""
        print("\n📊 INFORMATIONS DÉTAILLÉES MOVIELENS 100K")
        print("=" * 50)
        
        if self.ratings is not None:
            print("📈 DONNÉES DE NOTATIONS:")
            print(f"   • Nombre total de notations: {len(self.ratings):,}")
            print(f"   • Utilisateurs uniques: {self.ratings['user_id'].nunique()}")
            print(f"   • Films uniques: {self.ratings['movie_id'].nunique()}")
            print(f"   • Période: {pd.to_datetime(self.ratings['timestamp'].min(), unit='s').strftime('%Y-%m-%d')} to {pd.to_datetime(self.ratings['timestamp'].max(), unit='s').strftime('%Y-%m-%d')}")
            
            # Distribution des notes
            rating_stats = self.ratings['rating'].value_counts().sort_index()
            print(f"   • Distribution des notes:")
            for rating, count in rating_stats.items():
                percentage = (count / len(self.ratings)) * 100
                print(f"        {rating} étoiles: {count:>6} ({percentage:5.1f}%)")
        
        if self.movies is not None:
            print(f"\n🎬 DONNÉES DE FILMS:")
            print(f"   • Nombre total de films: {len(self.movies)}")
            if 'release_date' in self.movies.columns:
                valid_dates = pd.to_datetime(self.movies['release_date'], errors='coerce')
                year_counts = valid_dates.dt.year.value_counts().sort_index()
                print(f"   • Années de sortie: {year_counts.index.min()} - {year_counts.index.max()}")
            
            # Genres
            if 'Action' in self.movies.columns:
                genre_columns = self.movies.columns[6:24]  # Colonnes de genres
                genre_counts = self.movies[genre_columns].sum().sort_values(ascending=False)
                print(f"   • Top 5 genres:")
                for genre, count in genre_counts.head(5).items():
                    print(f"        {genre}: {count} films")
        