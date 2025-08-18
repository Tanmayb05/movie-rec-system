import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pickle
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class ContentBasedMovieRecommender:
    """
    Content-Based Filtering using multiple algorithms for movie recommendation
    """
    
    def __init__(self):
        self.movies_info = None
        self.movie_ratings = None
        self.movies_metadata = None
        
        # Trained models and vectorizers
        self.tfidf_overview = None
        self.tfidf_genres = None
        self.tfidf_keywords = None
        self.content_similarity_matrix = None
        self.scaler = None
        
        # Feature matrices
        self.content_features = None
        self.text_features = None
        self.numerical_features = None
        
    def load_data(self):
        """Load the movie datasets"""
        try:
            self.movies_info = pd.read_csv('data/movies_info.csv')
            self.movie_ratings = pd.read_csv('data/movies_ratings.csv')
            self.movies_metadata = pd.read_csv('data/movies_metadata.csv')
            
            print(f"✅ Loaded datasets:")
            print(f"   Movies Info: {self.movies_info.shape}")
            print(f"   Movie Ratings: {self.movie_ratings.shape}")
            print(f"   Movies Metadata: {self.movies_metadata.shape}")
            
            # Merge datasets
            self.merged_data = self.movies_info.merge(
                self.movie_ratings[['movie_id', 'vote_average', 'vote_count', 'popularity']], 
                on='movie_id', how='left'
            ).merge(
                self.movies_metadata[['movie_id', 'keywords', 'homepage']], 
                on='movie_id', how='left'
            )
            
            print(f"   Merged Dataset: {self.merged_data.shape}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    # ================================================================
    # ALGORITHM 1: TF-IDF + COSINE SIMILARITY (Text-Based)
    # ================================================================
    
    def build_tfidf_content_model(self):
        """
        Algorithm 1: TF-IDF Vectorization + Cosine Similarity
        Use Case: Text similarity based on overview, genres, keywords
        """
        print("\n🔤 BUILDING TF-IDF CONTENT MODEL")
        print("=" * 50)
        
        # Prepare text features
        self.merged_data['overview'] = self.merged_data['overview'].fillna('')
        self.merged_data['genres'] = self.merged_data['genres'].fillna('')
        self.merged_data['keywords'] = self.merged_data['keywords'].fillna('')
        
        # Combine text features
        self.merged_data['combined_text'] = (
            self.merged_data['overview'] + ' ' + 
            self.merged_data['genres'].str.replace('|', ' ') + ' ' +
            self.merged_data['keywords'].str.replace('|', ' ')
        )
        
        # TF-IDF Vectorization
        self.tfidf_overview = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        tfidf_matrix = self.tfidf_overview.fit_transform(self.merged_data['combined_text'])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print(f"✅ TF-IDF Model built:")
        print(f"   Feature matrix shape: {tfidf_matrix.shape}")
        print(f"   Similarity matrix shape: {self.content_similarity_matrix.shape}")
        print(f"   Vocabulary size: {len(self.tfidf_overview.vocabulary_)}")
        
        return tfidf_matrix
    
    def get_tfidf_recommendations(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using TF-IDF + Cosine Similarity"""
        
        try:
            # Find movie index
            movie_idx = self.merged_data[self.merged_data['movie_id'] == movie_id].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity_matrix[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar movies (excluding the movie itself)
            similar_movies = sim_scores[1:n_recommendations+1]
            
            recommendations = []
            for idx, score in similar_movies:
                movie_data = self.merged_data.iloc[idx]
                recommendations.append({
                    'movie_id': movie_data['movie_id'],
                    'title': movie_data['title'],
                    'similarity_score': score,
                    'genres': movie_data['genres'],
                    'vote_average': movie_data['vote_average'],
                    'overview': movie_data['overview'][:100] + '...'
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting TF-IDF recommendations: {e}")
            return []
    
    # ================================================================
    # ALGORITHM 2: FEATURE-BASED SIMILARITY (Numerical + Categorical)
    # ================================================================
    
    def build_feature_based_model(self):
        """
        Algorithm 2: Feature-based similarity using numerical and categorical features
        Use Case: Similarity based on runtime, budget, genres, language, etc.
        """
        print("\n📊 BUILDING FEATURE-BASED MODEL")
        print("=" * 50)
        
        # Prepare numerical features
        numerical_cols = ['runtime', 'budget', 'revenue', 'vote_average', 'vote_count', 'popularity']
        self.numerical_features = self.merged_data[numerical_cols].fillna(0)
        
        # Normalize numerical features
        self.scaler = StandardScaler()
        numerical_scaled = self.scaler.fit_transform(self.numerical_features)
        
        # Prepare categorical features (one-hot encoding for genres)
        genre_features = self.create_genre_features()
        language_features = self.create_language_features()
        
        # Combine all features
        self.content_features = np.hstack([
            numerical_scaled,
            genre_features,
            language_features
        ])
        
        print(f"✅ Feature-based Model built:")
        print(f"   Numerical features: {numerical_scaled.shape[1]}")
        print(f"   Genre features: {genre_features.shape[1]}")
        print(f"   Language features: {language_features.shape[1]}")
        print(f"   Total feature dimensions: {self.content_features.shape[1]}")
        
        return self.content_features
    
    def create_genre_features(self):
        """Create one-hot encoded genre features"""
        # Get all unique genres
        all_genres = set()
        for genres_str in self.merged_data['genres'].dropna():
            genres = genres_str.split('|')
            all_genres.update(genres)
        
        all_genres = sorted(list(all_genres))
        
        # Create genre matrix
        genre_matrix = np.zeros((len(self.merged_data), len(all_genres)))
        
        for i, genres_str in enumerate(self.merged_data['genres'].fillna('')):
            if genres_str:
                genres = genres_str.split('|')
                for genre in genres:
                    if genre in all_genres:
                        genre_idx = all_genres.index(genre)
                        genre_matrix[i, genre_idx] = 1
        
        self.genre_names = all_genres
        return genre_matrix
    
    def create_language_features(self):
        """Create one-hot encoded language features"""
        # Get top languages
        top_languages = self.merged_data['original_language'].value_counts().head(10).index.tolist()
        
        language_matrix = np.zeros((len(self.merged_data), len(top_languages)))
        
        for i, lang in enumerate(self.merged_data['original_language'].fillna('')):
            if lang in top_languages:
                lang_idx = top_languages.index(lang)
                language_matrix[i, lang_idx] = 1
        
        return language_matrix
    
    def get_feature_based_recommendations(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using feature-based similarity"""
        
        try:
            # Find movie index
            movie_idx = self.merged_data[self.merged_data['movie_id'] == movie_id].index[0]
            
            # Calculate cosine similarity with all movies
            target_features = self.content_features[movie_idx].reshape(1, -1)
            similarities = cosine_similarity(target_features, self.content_features)[0]
            
            # Get top similar movies
            similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
            
            recommendations = []
            for idx in similar_indices:
                movie_data = self.merged_data.iloc[idx]
                recommendations.append({
                    'movie_id': movie_data['movie_id'],
                    'title': movie_data['title'],
                    'similarity_score': similarities[idx],
                    'genres': movie_data['genres'],
                    'runtime': movie_data['runtime'],
                    'vote_average': movie_data['vote_average'],
                    'original_language': movie_data['original_language']
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting feature-based recommendations: {e}")
            return []
    
    # ================================================================
    # ALGORITHM 3: WEIGHTED HYBRID CONTENT MODEL
    # ================================================================
    
    def build_hybrid_content_model(self, text_weight: float = 0.6, feature_weight: float = 0.4):
        """
        Algorithm 3: Weighted combination of text-based and feature-based similarity
        Use Case: Best of both worlds - text understanding + numerical features
        """
        print(f"\n🔄 BUILDING HYBRID CONTENT MODEL (Text: {text_weight}, Features: {feature_weight})")
        print("=" * 60)
        
        # Ensure both models are built
        if self.content_similarity_matrix is None:
            self.build_tfidf_content_model()
        
        if self.content_features is None:
            self.build_feature_based_model()
        
        # Calculate feature-based similarity matrix
        feature_similarity = cosine_similarity(self.content_features)
        
        # Combine similarities
        self.hybrid_similarity = (
            text_weight * self.content_similarity_matrix +
            feature_weight * feature_similarity
        )
        
        print(f"✅ Hybrid model built with combined similarity matrix: {self.hybrid_similarity.shape}")
        
        return self.hybrid_similarity
    
    def get_hybrid_recommendations(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using hybrid content model"""
        
        try:
            # Find movie index
            movie_idx = self.merged_data[self.merged_data['movie_id'] == movie_id].index[0]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.hybrid_similarity[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar movies
            similar_movies = sim_scores[1:n_recommendations+1]
            
            recommendations = []
            for idx, score in similar_movies:
                movie_data = self.merged_data.iloc[idx]
                recommendations.append({
                    'movie_id': movie_data['movie_id'],
                    'title': movie_data['title'],
                    'similarity_score': score,
                    'genres': movie_data['genres'],
                    'vote_average': movie_data['vote_average'],
                    'runtime': movie_data['runtime'],
                    'original_language': movie_data['original_language']
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting hybrid recommendations: {e}")
            return []
    
    # ================================================================
    # ALGORITHM 4: CLUSTERING-BASED RECOMMENDATIONS
    # ================================================================
    
    def build_clustering_model(self, n_clusters: int = 50):
        """
        Algorithm 4: K-Means clustering for content-based recommendations
        Use Case: Group similar movies and recommend from same cluster
        """
        print(f"\n🎯 BUILDING CLUSTERING MODEL ({n_clusters} clusters)")
        print("=" * 50)
        
        # Use combined features for clustering
        if self.content_features is None:
            self.build_feature_based_model()
        
        # Apply PCA for dimensionality reduction (ensure n_components <= n_features)
        n_features = self.content_features.shape[1]
        n_components = min(30, n_features - 1)  # Use fewer components than features
        
        print(f"   Original features: {n_features}")
        print(f"   PCA components: {n_components}")
        
        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(self.content_features)
        
        # K-Means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.movie_clusters = self.kmeans.fit_predict(features_pca)
        
        # Add cluster labels to data
        self.merged_data['cluster'] = self.movie_clusters
        
        print(f"✅ Clustering model built:")
        print(f"   Number of clusters: {n_clusters}")
        print(f"   Original features: {n_features}")
        print(f"   PCA components: {features_pca.shape[1]}")
        print(f"   Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"   Movies per cluster (avg): {len(self.merged_data) / n_clusters:.1f}")
        
        # Cluster statistics
        cluster_counts = pd.Series(self.movie_clusters).value_counts().sort_index()
        print(f"   Cluster size range: {cluster_counts.min()} - {cluster_counts.max()} movies")
        
        return self.movie_clusters
    
    def get_clustering_recommendations(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using clustering"""
        
        try:
            # Find movie's cluster
            movie_row = self.merged_data[self.merged_data['movie_id'] == movie_id]
            if len(movie_row) == 0:
                return []
            
            movie_cluster = movie_row['cluster'].iloc[0]
            
            # Get all movies in the same cluster
            cluster_movies = self.merged_data[
                (self.merged_data['cluster'] == movie_cluster) & 
                (self.merged_data['movie_id'] != movie_id)
            ].copy()
            
            # Sort by rating and popularity within cluster
            cluster_movies['combined_score'] = (
                cluster_movies['vote_average'] * 0.7 + 
                np.log1p(cluster_movies['popularity']) * 0.3
            )
            
            top_cluster_movies = cluster_movies.nlargest(n_recommendations, 'combined_score')
            
            recommendations = []
            for _, movie_data in top_cluster_movies.iterrows():
                recommendations.append({
                    'movie_id': movie_data['movie_id'],
                    'title': movie_data['title'],
                    'cluster_id': movie_cluster,
                    'combined_score': movie_data['combined_score'],
                    'genres': movie_data['genres'],
                    'vote_average': movie_data['vote_average'],
                    'popularity': movie_data['popularity']
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting clustering recommendations: {e}")
            return []
    
    # ================================================================
    # ALGORITHM 5: MATRIX FACTORIZATION (Content-Based)
    # ================================================================
    
    def build_content_matrix_factorization(self, n_components: int = 30):
        """
        Algorithm 5: Matrix Factorization for content features
        Use Case: Dimensionality reduction and latent feature discovery
        """
        print(f"\n🎭 BUILDING CONTENT MATRIX FACTORIZATION ({n_components} components)")
        print("=" * 60)
        
        if self.content_features is None:
            self.build_feature_based_model()
        
        # Ensure n_components is valid
        n_features = self.content_features.shape[1]
        n_components = min(n_components, n_features - 1)
        
        print(f"   Original features: {n_features}")
        print(f"   Using components: {n_components}")
        
        # Use TruncatedSVD for matrix factorization
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.content_embeddings = self.svd_model.fit_transform(self.content_features)
        
        print(f"✅ Matrix Factorization model built:")
        print(f"   Original dimensions: {self.content_features.shape[1]}")
        print(f"   Reduced dimensions: {self.content_embeddings.shape[1]}")
        print(f"   Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.3f}")
        
        return self.content_embeddings
    
    def get_matrix_factorization_recommendations(self, movie_id: int, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using matrix factorization embeddings"""
        
        try:
            # Find movie index
            movie_idx = self.merged_data[self.merged_data['movie_id'] == movie_id].index[0]
            
            # Calculate cosine similarity in embedding space
            target_embedding = self.content_embeddings[movie_idx].reshape(1, -1)
            similarities = cosine_similarity(target_embedding, self.content_embeddings)[0]
            
            # Get top similar movies
            similar_indices = similarities.argsort()[::-1][1:n_recommendations+1]
            
            recommendations = []
            for idx in similar_indices:
                movie_data = self.merged_data.iloc[idx]
                recommendations.append({
                    'movie_id': movie_data['movie_id'],
                    'title': movie_data['title'],
                    'similarity_score': similarities[idx],
                    'genres': movie_data['genres'],
                    'vote_average': movie_data['vote_average'],
                    'embedding_dims': self.content_embeddings.shape[1]
                })
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error getting matrix factorization recommendations: {e}")
            return []
    
    # ================================================================
    # ALGORITHM COMPARISON AND EVALUATION
    # ================================================================
    
    def compare_algorithms(self, test_movie_id: int = 550, n_recs: int = 5):
        """
        Compare all content-based algorithms for a test movie
        """
        print(f"\n🔬 ALGORITHM COMPARISON FOR MOVIE ID: {test_movie_id}")
        print("=" * 60)
        
        # Get movie title
        movie_title = self.merged_data[self.merged_data['movie_id'] == test_movie_id]['title'].iloc[0]
        print(f"Target Movie: {movie_title}")
        
        algorithms = [
            ("TF-IDF + Cosine", self.get_tfidf_recommendations),
            ("Feature-Based", self.get_feature_based_recommendations),
            ("Hybrid Content", self.get_hybrid_recommendations),
            ("Clustering", self.get_clustering_recommendations),
            ("Matrix Factorization", self.get_matrix_factorization_recommendations)
        ]
        
        results = {}
        
        for algo_name, algo_func in algorithms:
            print(f"\n📊 {algo_name} Recommendations:")
            try:
                recs = algo_func(test_movie_id, n_recs)
                results[algo_name] = recs
                
                for i, rec in enumerate(recs, 1):
                    score_key = 'similarity_score' if 'similarity_score' in rec else 'combined_score'
                    score = rec.get(score_key, 0)
                    print(f"   {i}. {rec['title']} (Score: {score:.3f})")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[algo_name] = []
        
        return results
    
    def save_models(self, filename: str = 'content_based_models.pkl'):
        """Save trained models"""
        models = {
            'tfidf_overview': self.tfidf_overview,
            'content_similarity_matrix': self.content_similarity_matrix,
            'scaler': self.scaler,
            'content_features': self.content_features,
            'hybrid_similarity': getattr(self, 'hybrid_similarity', None),
            'kmeans': getattr(self, 'kmeans', None),
            'svd_model': getattr(self, 'svd_model', None),
            'content_embeddings': getattr(self, 'content_embeddings', None),
            'genre_names': getattr(self, 'genre_names', [])
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"✅ Models saved to {filename}")
    
    def load_models(self, filename: str = 'content_based_models.pkl'):
        """Load pre-trained models"""
        try:
            with open(filename, 'rb') as f:
                models = pickle.load(f)
            
            self.tfidf_overview = models['tfidf_overview']
            self.content_similarity_matrix = models['content_similarity_matrix']
            self.scaler = models['scaler']
            self.content_features = models['content_features']
            self.hybrid_similarity = models['hybrid_similarity']
            self.kmeans = models['kmeans']
            self.svd_model = models['svd_model']
            self.content_embeddings = models['content_embeddings']
            self.genre_names = models['genre_names']
            
            print(f"✅ Models loaded from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Model file {filename} not found")
            return False

def demo_content_based_filtering():
    """
    Demonstrate all content-based filtering algorithms
    """
    print("🎬 CONTENT-BASED FILTERING ALGORITHMS DEMO")
    print("=" * 60)
    
    # Initialize recommender
    recommender = ContentBasedMovieRecommender()
    
    # Load data
    if not recommender.load_data():
        print("❌ Could not load data. Please ensure CSV files are available.")
        return
    
    # Build all models
    print("\n🏗️ BUILDING ALL CONTENT-BASED MODELS...")
    
    # 1. TF-IDF Model
    recommender.build_tfidf_content_model()
    
    # 2. Feature-based Model  
    recommender.build_feature_based_model()
    
    # 3. Hybrid Model
    recommender.build_hybrid_content_model()
    
    # 4. Clustering Model
    recommender.build_clustering_model()
    
    # 5. Matrix Factorization Model
    recommender.build_content_matrix_factorization()
    
    # Compare algorithms
    test_movie_id = recommender.merged_data['movie_id'].iloc[0]  # First movie in dataset
    results = recommender.compare_algorithms(test_movie_id, n_recs=5)
    
    # Save models
    recommender.save_models()
    
    print(f"\n🎉 Content-based filtering demo completed!")
    print(f"All algorithms trained and ready for recommendations!")
    
    return recommender

if __name__ == "__main__":
    demo_content_based_filtering()