import pandas as pd
import numpy as np
import os
import ast
import re
from typing import List, Dict, Any
import kaggle
from dotenv import load_dotenv

class TMDBDataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.movies_df = None
        self.credits_df = None
        self.processed_df = None
        self._setup_kaggle_credentials()
        
    def _setup_kaggle_credentials(self):
        """Setup Kaggle credentials from .env file"""
        # Load environment variables from .env file
        load_dotenv()
        
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        
        if kaggle_username and kaggle_key:
            # Set environment variables for kaggle library
            os.environ['KAGGLE_USERNAME'] = kaggle_username
            os.environ['KAGGLE_KEY'] = kaggle_key
            print("✅ Kaggle credentials loaded from .env file")
        else:
            print("⚠️ Kaggle credentials not found in .env file, checking default location...")
    
    def download_dataset(self):
        """Download TMDB dataset from Kaggle"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Download the dataset
            print("Downloading TMDB dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                'tmdb/tmdb-movie-metadata', 
                path=self.data_dir, 
                unzip=True
            )
            print("Dataset downloaded successfully!")
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you have:")
            print("1. Kaggle API token configured")
            print("2. Internet connection")
            print("3. Kaggle account with API access")
    
    def load_data(self) -> tuple:
        """Load the TMDB dataset"""
        try:
            movies_path = os.path.join(self.data_dir, "tmdb_5000_movies.csv")
            credits_path = os.path.join(self.data_dir, "tmdb_5000_credits.csv")
            
            if not os.path.exists(movies_path) or not os.path.exists(credits_path):
                print("Dataset files not found. Downloading...")
                self.download_dataset()
            
            print("Loading TMDB dataset...")
            self.movies_df = pd.read_csv(movies_path)
            self.credits_df = pd.read_csv(credits_path)
            
            print(f"Loaded {len(self.movies_df)} movies and {len(self.credits_df)} credit records")
            return self.movies_df, self.credits_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def safe_eval(self, val):
        """Safely evaluate string representations of lists/dicts"""
        if pd.isna(val) or val == '':
            return []
        try:
            return ast.literal_eval(val)
        except:
            return []
    
    def extract_names(self, obj_list: List[Dict], key: str = 'name', limit: int = None) -> List[str]:
        """Extract names from list of dictionaries"""
        if not obj_list:
            return []
        names = [obj[key] for obj in obj_list if key in obj]
        return names[:limit] if limit else names
    
    def extract_cast_names(self, cast_list: List[Dict], limit: int = 5) -> List[str]:
        """Extract cast names with order consideration"""
        if not cast_list:
            return []
        # Sort by order if available, otherwise use original order
        try:
            sorted_cast = sorted(cast_list, key=lambda x: x.get('order', 999))
        except:
            sorted_cast = cast_list
        return [actor['name'] for actor in sorted_cast[:limit] if 'name' in actor]
    
    def clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text):
            return ""
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data for recommendation algorithms"""
        if self.movies_df is None or self.credits_df is None:
            self.load_data()
        
        print("Preprocessing data...")
        
        # Check column names and fix merge key
        print("Movies columns:", list(self.movies_df.columns))
        print("Credits columns:", list(self.credits_df.columns))
        
        # Find the correct ID column names
        movies_id_col = None
        credits_id_col = None
        
        for col in ['id', 'movie_id', 'tmdb_id']:
            if col in self.movies_df.columns:
                movies_id_col = col
                break
        
        for col in ['id', 'movie_id', 'tmdb_id']:
            if col in self.credits_df.columns:
                credits_id_col = col
                break
        
        if movies_id_col is None or credits_id_col is None:
            print(f"Warning: Could not find matching ID columns")
            print(f"Movies ID column: {movies_id_col}")
            print(f"Credits ID column: {credits_id_col}")
            # Use movies data only if merge fails
            df = self.movies_df.copy()
            # Add empty cast and crew columns
            df['cast'] = '[]'
            df['crew'] = '[]'
        else:
            # Merge movies and credits
            if movies_id_col != credits_id_col:
                # Rename to match
                self.credits_df = self.credits_df.rename(columns={credits_id_col: movies_id_col})
            
            # Drop title from credits to avoid duplicate columns
            credits_cols = [col for col in self.credits_df.columns if col != 'title']
            df = self.movies_df.merge(self.credits_df[credits_cols], on=movies_id_col, how='left')
        
        # Clean and extract features
        df['genres_list'] = df['genres'].apply(self.safe_eval).apply(lambda x: self.extract_names(x))
        df['keywords_list'] = df.get('keywords', pd.Series(['[]'] * len(df))).apply(self.safe_eval).apply(lambda x: self.extract_names(x, limit=10))
        df['cast_list'] = df.get('cast', pd.Series(['[]'] * len(df))).apply(self.safe_eval).apply(lambda x: self.extract_cast_names(x, limit=5))
        df['crew_list'] = df.get('crew', pd.Series(['[]'] * len(df))).apply(self.safe_eval)
        
        # Extract director
        df['director'] = df['crew_list'].apply(
            lambda x: [person['name'] for person in x if person.get('job') == 'Director'][:1] if x else []
        ).apply(lambda x: x[0] if x else "")
        
        # Extract production companies
        df['production_companies_list'] = df.get('production_companies', pd.Series(['[]'] * len(df))).apply(self.safe_eval).apply(
            lambda x: self.extract_names(x, limit=3)
        )
        
        # Clean text fields
        df['overview_clean'] = df['overview'].apply(self.clean_text)
        df['tagline_clean'] = df['tagline'].apply(self.clean_text)
        
        # Create combined text features
        df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join(x))
        df['keywords_str'] = df['keywords_list'].apply(lambda x: ' '.join(x))
        df['cast_str'] = df['cast_list'].apply(lambda x: ' '.join(x))
        df['companies_str'] = df['production_companies_list'].apply(lambda x: ' '.join(x))
        
        # Combined content for TF-IDF
        df['content'] = (
            df['overview_clean'] + ' ' + 
            df['tagline_clean'] + ' ' + 
            df['genres_str'] + ' ' + 
            df['keywords_str']
        )
        
        # Extract year from release_date
        df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
        
        # Handle missing values
        df = df.dropna(subset=['title', 'overview'])
        df = df.fillna('')
        
        # Normalize numerical features
        numerical_features = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count']
        for feature in numerical_features:
            if feature in df.columns:
                df[f'{feature}_norm'] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min() + 1e-8)
        
        # Select relevant columns
        relevant_columns = [
            'id', 'title', 'overview', 'tagline', 'release_date', 'release_year',
            'genres_list', 'keywords_list', 'cast_list', 'director',
            'production_companies_list', 'budget', 'revenue', 'runtime',
            'vote_average', 'vote_count', 'popularity',
            'genres_str', 'keywords_str', 'cast_str', 'companies_str', 'content',
            'budget_norm', 'revenue_norm', 'runtime_norm', 'vote_average_norm', 'vote_count_norm'
        ]
        
        # Use the actual ID column name if different
        if movies_id_col and movies_id_col != 'id':
            relevant_columns[0] = movies_id_col
        
        available_columns = [col for col in relevant_columns if col in df.columns]
        df_processed = df[available_columns].copy()
        
        # Remove duplicates
        df_processed = df_processed.drop_duplicates(subset=['title'])
        df_processed = df_processed.reset_index(drop=True)
        
        self.processed_df = df_processed
        
        print(f"Preprocessing complete. Final dataset shape: {df_processed.shape}")
        return df_processed
    
    def get_movie_titles(self) -> List[str]:
        """Get list of all movie titles"""
        if self.processed_df is None:
            self.preprocess_data()
        return self.processed_df['title'].tolist()
    
    def find_movie_index(self, title: str) -> int:
        """Find movie index by title"""
        if self.processed_df is None:
            self.preprocess_data()
        
        # Exact match first
        exact_match = self.processed_df[self.processed_df['title'].str.lower() == title.lower()]
        if not exact_match.empty:
            return exact_match.index[0]
        
        # Partial match
        partial_match = self.processed_df[
            self.processed_df['title'].str.lower().str.contains(title.lower(), na=False)
        ]
        if not partial_match.empty:
            return partial_match.index[0]
        
        return -1
    
    def get_movie_info(self, indices: List[int]) -> pd.DataFrame:
        """Get movie information by indices"""
        if self.processed_df is None:
            self.preprocess_data()
        
        return self.processed_df.iloc[indices][
            ['title', 'release_year', 'genres_str', 'vote_average', 'overview']
        ].copy()

# Example usage and testing
if __name__ == "__main__":
    # Initialize data loader
    loader = TMDBDataLoader()
    
    # Load and preprocess data
    df = loader.preprocess_data()
    
    # Display sample data
    print("\nSample of processed data:")
    print(df[['title', 'release_year', 'genres_str', 'vote_average']].head())
    
    # Test movie search
    test_movies = ["Avatar", "Inception", "The Dark Knight"]
    print("\nTesting movie search:")
    for movie in test_movies:
        idx = loader.find_movie_index(movie)
        if idx >= 0:
            print(f"'{movie}' found at index: {idx}")
        else:
            print(f"'{movie}' not found")