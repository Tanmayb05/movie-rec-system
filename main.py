#!/usr/bin/env python3
"""
TMDB Movie Recommendation System
Comprehensive content-based filtering with multiple algorithms
"""

import os
import sys
import time
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Load environment variables at the start
from dotenv import load_dotenv
load_dotenv()

# Import our custom modules
from data_loader import TMDBDataLoader
from content_algorithms import ContentBasedRecommender
from visualizer import RecommendationVisualizer

class MovieRecommendationSystem:
    def __init__(self):
        self.data_loader = TMDBDataLoader()
        self.df = None
        self.recommender = None
        self.visualizer = None
        
    def setup(self):
        """Initialize the recommendation system"""
        print("🎬 TMDB Movie Recommendation System")
        print("=" * 50)
        
        # Load and preprocess data
        print("\n📊 Loading and preprocessing data...")
        self.df = self.data_loader.preprocess_data()
        
        if self.df is None or self.df.empty:
            print("❌ Failed to load data. Please check your setup.")
            return False
        
        # Initialize recommender and visualizer
        self.recommender = ContentBasedRecommender(self.df)
        self.visualizer = RecommendationVisualizer(self.df)
        
        print(f"✅ System ready! Loaded {len(self.df)} movies")
        return True
    
    def get_user_input(self) -> List[str]:
        """Get movie preferences from user"""
        print("\n🎯 Enter 5 movies you like (one per line):")
        print("Tip: Use partial names if exact match doesn't work")
        print("-" * 40)
        
        movies = []
        for i in range(5):
            while True:
                movie = input(f"Movie {i+1}: ").strip()
                if movie:
                    movies.append(movie)
                    break
                else:
                    print("Please enter a valid movie name.")
        
        return movies
    
    def validate_movies(self, input_movies: List[str]) -> Tuple[List[str], List[int]]:
        """Validate and find movie indices"""
        found_movies = []
        movie_indices = []
        
        print("\n🔍 Searching for movies in database...")
        
        for movie in input_movies:
            idx = self.data_loader.find_movie_index(movie)
            if idx >= 0:
                actual_title = self.df.iloc[idx]['title']
                found_movies.append(actual_title)
                movie_indices.append(idx)
                print(f"✅ Found: '{movie}' → '{actual_title}'")
            else:
                print(f"❌ Not found: '{movie}'")
                # Suggest similar movies
                self.suggest_similar_movies(movie)
        
        return found_movies, movie_indices
    
    def suggest_similar_movies(self, movie_name: str, top_n: int = 3):
        """Suggest similar movie names"""
        from fuzzywuzzy import fuzz
        
        similarities = []
        for idx, title in enumerate(self.df['title']):
            similarity = fuzz.partial_ratio(movie_name.lower(), title.lower())
            if similarity > 60:  # Threshold for suggestion
                similarities.append((idx, title, similarity))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        if similarities:
            print(f"   📝 Did you mean: {', '.join([title for _, title, _ in similarities[:top_n]])}")
    
    def run_recommendations(self, movie_indices: List[int], found_movies: List[str]) -> Dict:
        """Run all recommendation algorithms"""
        print(f"\n🚀 Running recommendation algorithms...")
        print(f"Based on: {', '.join(found_movies)}")
        
        start_time = time.time()
        
        # Run all algorithms
        results = self.recommender.run_all_algorithms(movie_indices, n_recommendations=10)
        
        end_time = time.time()
        print(f"\n⏱️ Total processing time: {end_time - start_time:.2f} seconds")
        
        return results
    
    def display_results(self, results: Dict, found_movies: List[str]):
        """Display recommendation results"""
        # Print detailed results
        self.visualizer.print_detailed_results(results, found_movies)
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        self.visualizer.create_comprehensive_report(results, found_movies)
    
    def interactive_mode(self):
        """Interactive mode for multiple recommendations"""
        while True:
            print("\n" + "="*50)
            print("🔄 INTERACTIVE MODE")
            print("="*50)
            
            choice = input("\nOptions:\n1. New recommendation\n2. Exit\nChoose (1-2): ").strip()
            
            if choice == '1':
                self.run_single_recommendation()
            elif choice == '2':
                print("👋 Thank you for using the Movie Recommendation System!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def run_single_recommendation(self):
        """Run a single recommendation cycle"""
        # Get user input
        input_movies = self.get_user_input()
        
        # Validate movies
        found_movies, movie_indices = self.validate_movies(input_movies)
        
        if len(movie_indices) == 0:
            print("❌ No movies found in database. Please try again with different titles.")
            return
        
        if len(movie_indices) < len(input_movies):
            proceed = input(f"\n⚠️ Only {len(movie_indices)} out of {len(input_movies)} movies found. Proceed? (y/n): ")
            if proceed.lower() != 'y':
                return
        
        # Run recommendations
        results = self.run_recommendations(movie_indices, found_movies)
        
        # Display results
        self.display_results(results, found_movies)
    
    def run(self):
        """Main execution method"""
        # Setup system
        if not self.setup():
            return
        
        # Display available movies sample
        self.show_sample_movies()
        
        # Run single recommendation
        self.run_single_recommendation()
        
        # Ask for interactive mode
        continue_choice = input("\n🔄 Would you like to try more recommendations? (y/n): ")
        if continue_choice.lower() == 'y':
            self.interactive_mode()
    
    def show_sample_movies(self):
        """Show sample of available movies"""
        print("\n🎬 Sample of available movies:")
        print("-" * 30)
        
        # Show random sample
        sample_movies = self.df.sample(10)[['title', 'release_year', 'vote_average']].copy()
        for _, movie in sample_movies.iterrows():
            year = int(movie['release_year']) if pd.notna(movie['release_year']) else 'N/A'
            rating = f"{movie['vote_average']:.1f}" if pd.notna(movie['vote_average']) else 'N/A'
            print(f"• {movie['title']} ({year}) - Rating: {rating}")
        
        print(f"\n📈 Total movies in database: {len(self.df)}")

def main():
    """Main function"""
    try:
        # Create and run the recommendation system
        system = MovieRecommendationSystem()
        system.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    # Import pandas here to avoid issues
    import pandas as pd
    main()