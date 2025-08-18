import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import fuzz
from typing import List, Tuple, Dict
import networkx as nx
from sklearn.cluster import KMeans
import warnings
import time
import logging

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.similarity_matrices = {}
        
    def log_time(self, start_time: float, algorithm_name: str, step: str = ""):
        """Log timing information"""
        elapsed = time.time() - start_time
        if step:
            logger.info(f"{algorithm_name} - {step}: {elapsed:.2f}s")
        else:
            logger.info(f"{algorithm_name} completed in: {elapsed:.2f}s")
        return elapsed
        
    def get_recommendations(self, movie_indices: List[int], similarity_matrix: np.ndarray, 
                          algorithm_name: str, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations based on similarity matrix"""
        # Calculate average similarity for input movies
        avg_similarity = np.mean(similarity_matrix[movie_indices], axis=0)
        
        # Remove input movies from recommendations
        for idx in movie_indices:
            avg_similarity[idx] = -1
        
        # Get top recommendations
        top_indices = avg_similarity.argsort()[-n_recommendations:][::-1]
        recommendations = [(idx, avg_similarity[idx]) for idx in top_indices]
        
        return recommendations
    
    def algorithm_1_tfidf_cosine(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 1: TF-IDF with Cosine Similarity"""
        start_time = time.time()
        logger.info("Starting Algorithm 1: TF-IDF with Cosine Similarity...")
        
        try:
            # Create TF-IDF matrix
            step_time = time.time()
            tfidf = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Use combined content (overview + genres + keywords)
            tfidf_matrix = tfidf.fit_transform(self.df['content'].fillna(''))
            self.log_time(step_time, "TF-IDF", "TF-IDF vectorization")
            
            # Calculate cosine similarity
            step_time = time.time()
            cosine_sim = cosine_similarity(tfidf_matrix)
            self.log_time(step_time, "TF-IDF", "Cosine similarity calculation")
            
            self.similarity_matrices['tfidf_cosine'] = cosine_sim
            self.log_time(start_time, "TF-IDF Algorithm")
            
            return self.get_recommendations(movie_indices, cosine_sim, "TF-IDF Cosine", n_recommendations)
            
        except Exception as e:
            logger.error(f"Error in TF-IDF algorithm: {str(e)}")
            return []
    
    def algorithm_2_count_vectorizer_cosine(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 2: Count Vectorizer with Cosine Similarity"""
        print("Running Algorithm 2: Count Vectorizer with Cosine Similarity...")
        
        # Create Count Vectorizer matrix
        count_vec = CountVectorizer(
            stop_words='english',
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            binary=True  # Binary occurrence
        )
        
        count_matrix = count_vec.fit_transform(self.df['content'].fillna(''))
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(count_matrix)
        self.similarity_matrices['count_cosine'] = cosine_sim
        
        return self.get_recommendations(movie_indices, cosine_sim, "Count Vector Cosine", n_recommendations)
    
    def algorithm_3_weighted_features(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 3: Weighted Multi-Feature Similarity"""
        start_time = time.time()
        logger.info("Starting Algorithm 3: Weighted Multi-Feature Similarity...")
        
        try:
            # Genre similarity (Jaccard)
            step_time = time.time()
            mlb_genres = MultiLabelBinarizer()
            genre_matrix = mlb_genres.fit_transform(self.df['genres_list'])
            genre_sim = cosine_similarity(genre_matrix)
            self.log_time(step_time, "Weighted Features", "Genre similarity")
            
            # Cast similarity
            step_time = time.time()
            mlb_cast = MultiLabelBinarizer()
            cast_matrix = mlb_cast.fit_transform(self.df['cast_list'])
            cast_sim = cosine_similarity(cast_matrix)
            self.log_time(step_time, "Weighted Features", "Cast similarity")
            
            # Keyword similarity
            step_time = time.time()
            mlb_keywords = MultiLabelBinarizer()
            keyword_matrix = mlb_keywords.fit_transform(self.df['keywords_list'])
            keyword_sim = cosine_similarity(keyword_matrix)
            self.log_time(step_time, "Weighted Features", "Keyword similarity")
            
            # Director similarity (exact match)
            step_time = time.time()
            directors = self.df['director'].fillna('').values  # Fix: handle NaN values
            director_sim = np.array([[1 if (d1 == d2 and d1 != '') else 0 for d2 in directors] for d1 in directors])
            self.log_time(step_time, "Weighted Features", "Director similarity")
            
            # Year proximity (normalized)
            step_time = time.time()
            years = self.df['release_year'].fillna(2000).values
            # Convert to numeric and handle any remaining non-numeric values
            years = pd.to_numeric(years, errors='coerce')
            years = np.nan_to_num(years, nan=2000.0)  # Replace NaN with 2000
            
            year_diff = np.abs(years[:, np.newaxis] - years)
            year_sim = np.exp(-year_diff / 5)  # Decay function
            self.log_time(step_time, "Weighted Features", "Year similarity")
            
            # Combined weighted similarity
            step_time = time.time()
            weighted_sim = (
                0.35 * genre_sim +
                0.25 * cast_sim +
                0.20 * keyword_sim +
                0.15 * director_sim +
                0.05 * year_sim
            )
            self.log_time(step_time, "Weighted Features", "Combination")
            
            self.similarity_matrices['weighted_features'] = weighted_sim
            self.log_time(start_time, "Weighted Features Algorithm")
            
            return self.get_recommendations(movie_indices, weighted_sim, "Weighted Features", n_recommendations)
            
        except Exception as e:
            logger.error(f"Error in Weighted Features algorithm: {str(e)}")
            return []
    
    def algorithm_4_euclidean_similarity(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 4: Euclidean Distance-Based Similarity"""
        print("Running Algorithm 4: Euclidean Distance-Based Similarity...")
        
        # Prepare numerical features
        numerical_features = ['budget_norm', 'revenue_norm', 'runtime_norm', 
                            'vote_average_norm', 'vote_count_norm']
        
        # Handle missing features gracefully
        available_features = [f for f in numerical_features if f in self.df.columns]
        if not available_features:
            print("Warning: No numerical features available, using basic features")
            feature_matrix = np.random.random((len(self.df), 3))  # Fallback
        else:
            feature_matrix = self.df[available_features].fillna(0).values
        
        # Calculate Euclidean distances
        distances = euclidean_distances(feature_matrix)
        
        # Convert distance to similarity (inverse relationship)
        max_distance = np.max(distances)
        euclidean_sim = 1 - (distances / max_distance)
        
        self.similarity_matrices['euclidean'] = euclidean_sim
        
        return self.get_recommendations(movie_indices, euclidean_sim, "Euclidean Similarity", n_recommendations)
    
    def algorithm_5_fuzzy_cast_similarity(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 5: Fuzzy String Matching for Cast/Crew"""
        start_time = time.time()
        logger.info("Starting Algorithm 5: Fuzzy String Matching...")
        
        try:
            def fuzzy_cast_similarity(cast1, cast2):
                if not cast1 or not cast2:
                    return 0
                
                max_sim = 0
                for actor1 in cast1:
                    for actor2 in cast2:
                        sim = fuzz.ratio(actor1.lower(), actor2.lower()) / 100.0
                        max_sim = max(max_sim, sim)
                
                # Also consider cast overlap
                overlap = len(set(cast1) & set(cast2)) / len(set(cast1) | set(cast2)) if cast1 or cast2 else 0
                
                return max(max_sim, overlap)
            
            # Calculate fuzzy similarity matrix
            n_movies = len(self.df)
            fuzzy_sim = np.zeros((n_movies, n_movies))
            
            cast_lists = self.df['cast_list'].tolist()
            
            # Optimize: only calculate for a sample if dataset is large
            sample_size = min(1000, n_movies)  # Limit for performance
            if n_movies > sample_size:
                logger.info(f"Large dataset detected. Processing sample of {sample_size} movies...")
                indices = np.random.choice(n_movies, sample_size, replace=False)
                cast_lists = [cast_lists[i] for i in indices]
                fuzzy_sim = np.zeros((sample_size, sample_size))
                n_movies = sample_size
            
            step_time = time.time()
            for i in range(n_movies):
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{n_movies} movies for fuzzy matching...")
                for j in range(i, n_movies):
                    if i == j:
                        fuzzy_sim[i, j] = 1.0
                    else:
                        sim = fuzzy_cast_similarity(cast_lists[i], cast_lists[j])
                        fuzzy_sim[i, j] = sim
                        fuzzy_sim[j, i] = sim
            
            self.log_time(step_time, "Fuzzy Cast", "Similarity matrix calculation")
            self.similarity_matrices['fuzzy_cast'] = fuzzy_sim
            self.log_time(start_time, "Fuzzy Cast Matching Algorithm")
            
            return self.get_recommendations(movie_indices, fuzzy_sim, "Fuzzy Cast Matching", n_recommendations)
            
        except Exception as e:
            logger.error(f"Error in Fuzzy Cast Matching algorithm: {str(e)}")
            return []
    
    def algorithm_6_categorical_similarity(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 6: Multi-Feature Categorical Similarity"""
        print("Running Algorithm 6: Multi-Feature Categorical Similarity...")
        
        def jaccard_similarity(set1, set2):
            if not set1 and not set2:
                return 1.0
            union = len(set1.union(set2))
            if union == 0:
                return 0.0
            intersection = len(set1.intersection(set2))
            return intersection / union
        
        # Calculate similarity matrix
        n_movies = len(self.df)
        categorical_sim = np.zeros((n_movies, n_movies))
        
        for i in range(n_movies):
            for j in range(i, n_movies):
                if i == j:
                    categorical_sim[i, j] = 1.0
                else:
                    # Genre Jaccard similarity
                    genres1 = set(self.df.iloc[i]['genres_list'])
                    genres2 = set(self.df.iloc[j]['genres_list'])
                    genre_sim = jaccard_similarity(genres1, genres2)
                    
                    # Keywords Jaccard similarity
                    keywords1 = set(self.df.iloc[i]['keywords_list'])
                    keywords2 = set(self.df.iloc[j]['keywords_list'])
                    keyword_sim = jaccard_similarity(keywords1, keywords2)
                    
                    # Cast overlap
                    cast1 = set(self.df.iloc[i]['cast_list'])
                    cast2 = set(self.df.iloc[j]['cast_list'])
                    cast_sim = jaccard_similarity(cast1, cast2)
                    
                    # Combined similarity
                    combined_sim = 0.5 * genre_sim + 0.3 * keyword_sim + 0.2 * cast_sim
                    
                    categorical_sim[i, j] = combined_sim
                    categorical_sim[j, i] = combined_sim
        
        self.similarity_matrices['categorical'] = categorical_sim
        
        return self.get_recommendations(movie_indices, categorical_sim, "Categorical Similarity", n_recommendations)
    
    def algorithm_7_clustering_based(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 7: Clustering-Based Recommendations"""
        print("Running Algorithm 7: Clustering-Based Similarity...")
        
        # Prepare features for clustering
        mlb_genres = MultiLabelBinarizer()
        genre_features = mlb_genres.fit_transform(self.df['genres_list'])
        
        mlb_cast = MultiLabelBinarizer()
        cast_features = mlb_cast.fit_transform(self.df['cast_list'])
        
        # Combine features
        feature_matrix = np.hstack([genre_features, cast_features[:, :20]])  # Limit cast features
        
        # Perform clustering
        n_clusters = min(50, len(self.df) // 10)  # Dynamic cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(feature_matrix)
        
        # Calculate cluster-based similarity
        n_movies = len(self.df)
        cluster_sim = np.zeros((n_movies, n_movies))
        
        for i in range(n_movies):
            for j in range(n_movies):
                if clusters[i] == clusters[j]:
                    # Within same cluster, use cosine similarity of features
                    cluster_sim[i, j] = cosine_similarity([feature_matrix[i]], [feature_matrix[j]])[0, 0]
                else:
                    # Different clusters have lower similarity
                    cluster_sim[i, j] = 0.1 * cosine_similarity([feature_matrix[i]], [feature_matrix[j]])[0, 0]
        
        self.similarity_matrices['clustering'] = cluster_sim
        
        return self.get_recommendations(movie_indices, cluster_sim, "Clustering-Based", n_recommendations)
    
    def algorithm_8_graph_based(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 8: Graph-Based Recommendations"""
        start_time = time.time()
        logger.info("Starting Algorithm 8: Graph-Based Similarity...")
        
        try:
            # Create a graph where movies are nodes
            step_time = time.time()
            G = nx.Graph()
            
            # Optimize: limit graph size for performance
            n_movies = min(1000, len(self.df))  # Limit for performance
            logger.info(f"Creating graph with {n_movies} movies...")
            
            # Add nodes (movies)
            for i in range(n_movies):
                title = self.df.iloc[i]['title']
                G.add_node(i, title=title)
            
            self.log_time(step_time, "Graph-Based", "Node creation")
            
            # Add edges based on shared features
            step_time = time.time()
            edge_count = 0
            for i in range(n_movies):
                if i % 100 == 0:
                    logger.info(f"Processing edges for movie {i}/{n_movies}...")
                
                for j in range(i + 1, min(i + 50, n_movies)):  # Limit edges per node
                    weight = 0
                    
                    # Shared genres
                    genres1 = set(self.df.iloc[i]['genres_list'])
                    genres2 = set(self.df.iloc[j]['genres_list'])
                    genre_overlap = len(genres1 & genres2)
                    weight += genre_overlap * 2
                    
                    # Shared cast
                    cast1 = set(self.df.iloc[i]['cast_list'])
                    cast2 = set(self.df.iloc[j]['cast_list'])
                    cast_overlap = len(cast1 & cast2)
                    weight += cast_overlap
                    
                    # Same director
                    if (self.df.iloc[i]['director'] == self.df.iloc[j]['director'] and 
                        self.df.iloc[i]['director'] != ''):
                        weight += 3
                    
                    # Add edge if weight is significant
                    if weight > 0:
                        G.add_edge(i, j, weight=weight)
                        edge_count += 1
            
            logger.info(f"Created graph with {len(G.nodes)} nodes and {edge_count} edges")
            self.log_time(step_time, "Graph-Based", "Edge creation")
            
            # Calculate node similarities using adjacency matrix
            step_time = time.time()
            try:
                adj_matrix = nx.adjacency_matrix(G, weight='weight').toarray()
                graph_sim = cosine_similarity(adj_matrix)
            except:
                # Fallback: use simple adjacency
                adj_matrix = nx.adjacency_matrix(G).toarray()
                graph_sim = cosine_similarity(adj_matrix)
            
            self.log_time(step_time, "Graph-Based", "Similarity calculation")
            
            # Pad matrix if we used a subset
            if n_movies < len(self.df):
                full_sim = np.zeros((len(self.df), len(self.df)))
                full_sim[:n_movies, :n_movies] = graph_sim
                graph_sim = full_sim
            
            self.similarity_matrices['graph_based'] = graph_sim
            self.log_time(start_time, "Graph-Based Algorithm")
            
            return self.get_recommendations(movie_indices, graph_sim, "Graph-Based", n_recommendations)
            
        except Exception as e:
            logger.error(f"Error in Graph-Based algorithm: {str(e)}")
            return []
    
    def algorithm_9_temporal_content(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 9: Temporal Content-Based Filtering"""
        print("Running Algorithm 9: Temporal Content-Based...")
        
        # Base content similarity (TF-IDF)
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = tfidf.fit_transform(self.df['content'].fillna(''))
        content_sim = cosine_similarity(tfidf_matrix)
        
        # Temporal weights based on release year
        years = self.df['release_year'].fillna(2000).values
        current_year = 2024
        
        # Create temporal decay matrix
        temporal_weights = np.zeros((len(self.df), len(self.df)))
        
        for i in range(len(self.df)):
            for j in range(len(self.df)):
                year_diff = abs(years[i] - years[j])
                # Movies from same era get higher weights
                era_weight = np.exp(-year_diff / 10)
                
                # Recent movies get slight boost
                recency_weight1 = 1 + 0.1 * np.exp(-(current_year - years[i]) / 5)
                recency_weight2 = 1 + 0.1 * np.exp(-(current_year - years[j]) / 5)
                
                temporal_weights[i, j] = era_weight * np.sqrt(recency_weight1 * recency_weight2)
        
        # Combine content similarity with temporal weights
        temporal_sim = content_sim * temporal_weights
        
        self.similarity_matrices['temporal'] = temporal_sim
        
        return self.get_recommendations(movie_indices, temporal_sim, "Temporal Content", n_recommendations)
    
    def algorithm_10_hybrid_content(self, movie_indices: List[int], n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Algorithm 10: Hybrid Content-Based Approach"""
        print("Running Algorithm 10: Hybrid Content-Based...")
        
        # Ensure we have necessary similarity matrices
        if 'tfidf_cosine' not in self.similarity_matrices:
            self.algorithm_1_tfidf_cosine(movie_indices, n_recommendations)
        
        if 'weighted_features' not in self.similarity_matrices:
            self.algorithm_3_weighted_features(movie_indices, n_recommendations)
        
        if 'categorical' not in self.similarity_matrices:
            self.algorithm_6_categorical_similarity(movie_indices, n_recommendations)
        
        # Combine multiple approaches with weights
        hybrid_sim = (
            0.4 * self.similarity_matrices.get('tfidf_cosine', np.zeros((len(self.df), len(self.df)))) +
            0.3 * self.similarity_matrices.get('weighted_features', np.zeros((len(self.df), len(self.df)))) +
            0.2 * self.similarity_matrices.get('categorical', np.zeros((len(self.df), len(self.df)))) +
            0.1 * self.similarity_matrices.get('euclidean', np.eye(len(self.df)))
        )
        
        self.similarity_matrices['hybrid'] = hybrid_sim
        
        return self.get_recommendations(movie_indices, hybrid_sim, "Hybrid Content", n_recommendations)
    
    def run_all_algorithms(self, movie_indices: List[int], n_recommendations: int = 10) -> Dict[str, List[Tuple[int, float]]]:
        """Run all content-based algorithms"""
        results = {}
        timing_results = {}
        
        algorithms = [
            ('TF-IDF Cosine', self.algorithm_1_tfidf_cosine),
            ('Count Vector Cosine', self.algorithm_2_count_vectorizer_cosine),
            ('Weighted Features', self.algorithm_3_weighted_features),
            ('Euclidean Similarity', self.algorithm_4_euclidean_similarity),
            ('Fuzzy Cast Matching', self.algorithm_5_fuzzy_cast_similarity),
            ('Categorical Similarity', self.algorithm_6_categorical_similarity),
            ('Clustering-Based', self.algorithm_7_clustering_based),
            ('Graph-Based', self.algorithm_8_graph_based),
            ('Temporal Content', self.algorithm_9_temporal_content),
            ('Hybrid Content', self.algorithm_10_hybrid_content)
        ]
        
        total_start_time = time.time()
        
        for name, algorithm in algorithms:
            try:
                print(f"\n{'='*50}")
                print(f"Running {name}...")
                
                algo_start_time = time.time()
                recommendations = algorithm(movie_indices, n_recommendations)
                algo_time = time.time() - algo_start_time
                
                results[name] = recommendations
                timing_results[name] = algo_time
                
                print(f"✓ {name} completed in {algo_time:.2f}s ({len(recommendations)} recommendations)")
                
            except Exception as e:
                print(f"✗ Error in {name}: {str(e)}")
                logger.error(f"Error in {name}: {str(e)}")
                results[name] = []
                timing_results[name] = 0
        
        total_time = time.time() - total_start_time
        
        # Print timing summary
        print(f"\n{'='*50}")
        print("TIMING SUMMARY")
        print(f"{'='*50}")
        for name, exec_time in timing_results.items():
            if exec_time > 0:
                print(f"{name:<25}: {exec_time:>8.2f}s")
        print(f"{'='*50}")
        print(f"{'Total Time':<25}: {total_time:>8.2f}s")
        print(f"{'='*50}")
        
        return results