import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class RecommendationVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        plt.style.use('seaborn-v0_8')
        
    def plot_algorithm_comparison(self, results: Dict[str, List[Tuple[int, float]]], 
                                input_movies: List[str], save_path: str = None):
        """Plot comparison of different algorithms"""
        
        # Prepare data for plotting
        algorithm_names = list(results.keys())
        avg_similarities = []
        
        for algo_name, recommendations in results.items():
            if recommendations:
                similarities = [sim for _, sim in recommendations]
                avg_similarities.append(np.mean(similarities))
            else:
                avg_similarities.append(0)
        
        # Create bar plot
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(algorithm_names)), avg_similarities, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(algorithm_names))))
        
        plt.title(f'Algorithm Performance Comparison\nInput Movies: {", ".join(input_movies)}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Algorithms', fontsize=12)
        plt.ylabel('Average Similarity Score', fontsize=12)
        plt.xticks(range(len(algorithm_names)), algorithm_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, avg_similarities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_recommendation_heatmap(self, results: Dict[str, List[Tuple[int, float]]], 
                                  top_n: int = 5, save_path: str = None):
        """Create heatmap of top recommendations across algorithms"""
        
        # Get top movies for each algorithm
        all_movies = set()
        algo_recommendations = {}
        
        for algo_name, recommendations in results.items():
            top_movies = [self.df.iloc[idx]['title'] for idx, _ in recommendations[:top_n]]
            algo_recommendations[algo_name] = top_movies
            all_movies.update(top_movies)
        
        # Create binary matrix
        movie_list = list(all_movies)
        binary_matrix = np.zeros((len(algo_recommendations), len(movie_list)))
        
        for i, (algo_name, movies) in enumerate(algo_recommendations.items()):
            for movie in movies:
                j = movie_list.index(movie)
                binary_matrix[i, j] = 1
        
        # Create heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(binary_matrix, 
                   xticklabels=[movie[:30] + '...' if len(movie) > 30 else movie for movie in movie_list],
                   yticklabels=list(algo_recommendations.keys()),
                   cmap='Reds', 
                   cbar_kws={'label': 'Recommended'},
                   linewidths=0.5)
        
        plt.title('Recommendation Overlap Across Algorithms', fontsize=16, fontweight='bold')
        plt.xlabel('Movies', fontsize=12)
        plt.ylabel('Algorithms', fontsize=12)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_similarity_distribution(self, results: Dict[str, List[Tuple[int, float]]], 
                                   save_path: str = None):
        """Plot distribution of similarity scores for each algorithm"""
        
        plt.figure(figsize=(15, 10))
        
        for i, (algo_name, recommendations) in enumerate(results.items()):
            if recommendations:
                similarities = [sim for _, sim in recommendations]
                plt.subplot(2, 5, i + 1)
                plt.hist(similarities, bins=20, alpha=0.7, color=plt.cm.Set3(i/10))
                plt.title(f'{algo_name}')
                plt.xlabel('Similarity Score')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_comparison(self, results: Dict[str, List[Tuple[int, float]]], 
                                    input_movies: List[str], save_path: str = None):
        """Create interactive Plotly visualization"""
        
        # Prepare data
        plot_data = []
        
        for algo_name, recommendations in results.items():
            for rank, (movie_idx, similarity) in enumerate(recommendations[:10], 1):
                movie_info = self.df.iloc[movie_idx]
                plot_data.append({
                    'Algorithm': algo_name,
                    'Rank': rank,
                    'Movie': movie_info['title'],
                    'Similarity': similarity,
                    'Year': movie_info.get('release_year', 'N/A'),
                    'Rating': movie_info.get('vote_average', 'N/A'),
                    'Genres': movie_info.get('genres_str', 'N/A')
                })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create interactive scatter plot
        fig = px.scatter(df_plot, 
                        x='Rank', 
                        y='Similarity',
                        color='Algorithm',
                        hover_data=['Movie', 'Year', 'Rating', 'Genres'],
                        title=f'Interactive Recommendation Comparison<br>Input Movies: {", ".join(input_movies)}',
                        height=600)
        
        fig.update_layout(
            xaxis_title="Recommendation Rank",
            yaxis_title="Similarity Score",
            hovermode='closest'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_genre_distribution(self, results: Dict[str, List[Tuple[int, float]]], 
                              input_movies: List[str], save_path: str = None):
        """Plot genre distribution of recommendations"""
        
        # Get genres of input movies
        input_indices = []
        for movie in input_movies:
            for idx, title in enumerate(self.df['title']):
                if movie.lower() in title.lower():
                    input_indices.append(idx)
                    break
        
        input_genres = set()
        for idx in input_indices:
            input_genres.update(self.df.iloc[idx]['genres_list'])
        
        # Analyze recommended movies' genres
        genre_data = []
        
        for algo_name, recommendations in results.items():
            algo_genres = {}
            for movie_idx, _ in recommendations[:10]:
                movie_genres = self.df.iloc[movie_idx]['genres_list']
                for genre in movie_genres:
                    algo_genres[genre] = algo_genres.get(genre, 0) + 1
            
            for genre, count in algo_genres.items():
                genre_data.append({
                    'Algorithm': algo_name,
                    'Genre': genre,
                    'Count': count,
                    'Input_Genre': genre in input_genres
                })
        
        df_genre = pd.DataFrame(genre_data)
        
        if not df_genre.empty:
            # Create grouped bar plot
            plt.figure(figsize=(15, 8))
            
            # Get top genres
            top_genres = df_genre.groupby('Genre')['Count'].sum().nlargest(10).index
            df_genre_filtered = df_genre[df_genre['Genre'].isin(top_genres)]
            
            pivot_data = df_genre_filtered.pivot(index='Genre', columns='Algorithm', values='Count').fillna(0)
            
            ax = pivot_data.plot(kind='bar', figsize=(15, 8), width=0.8)
            plt.title('Genre Distribution in Recommendations', fontsize=16, fontweight='bold')
            plt.xlabel('Genres', fontsize=12)
            plt.ylabel('Count in Top 10 Recommendations', fontsize=12)
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_comprehensive_report(self, results: Dict[str, List[Tuple[int, float]]], 
                                  input_movies: List[str], output_dir: str = "visualizations"):
        """Create comprehensive visualization report"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating comprehensive visualization report...")
        
        # 1. Algorithm comparison
        self.plot_algorithm_comparison(
            results, input_movies, 
            save_path=f"{output_dir}/algorithm_comparison.png"
        )
        
        # 2. Recommendation heatmap
        self.plot_recommendation_heatmap(
            results, top_n=5, 
            save_path=f"{output_dir}/recommendation_heatmap.png"
        )
        
        # 3. Similarity distribution
        self.plot_similarity_distribution(
            results, 
            save_path=f"{output_dir}/similarity_distribution.png"
        )
        
        # 4. Interactive comparison
        self.create_interactive_comparison(
            results, input_movies, 
            save_path=f"{output_dir}/interactive_comparison.html"
        )
        
        # 5. Genre distribution
        self.plot_genre_distribution(
            results, input_movies, 
            save_path=f"{output_dir}/genre_distribution.png"
        )
        
        print(f"Visualization report saved to '{output_dir}' directory")
    
    def print_detailed_results(self, results: Dict[str, List[Tuple[int, float]]], 
                             input_movies: List[str]):
        """Print detailed results in a formatted table"""
        
        print("\n" + "="*80)
        print(f"MOVIE RECOMMENDATION RESULTS")
        print(f"Input Movies: {', '.join(input_movies)}")
        print("="*80)
        
        for algo_name, recommendations in results.items():
            print(f"\n{algo_name.upper()}")
            print("-" * 50)
            
            if not recommendations:
                print("No recommendations generated")
                continue
            
            print(f"{'Rank':<4} {'Title':<35} {'Year':<6} {'Rating':<7} {'Similarity':<10}")
            print("-" * 70)
            
            for rank, (movie_idx, similarity) in enumerate(recommendations, 1):
                movie = self.df.iloc[movie_idx]
                title = movie['title'][:32] + "..." if len(movie['title']) > 35 else movie['title']
                year = str(movie.get('release_year', 'N/A'))[:4]
                rating = f"{movie.get('vote_average', 0):.1f}"
                sim_score = f"{similarity:.3f}"
                
                print(f"{rank:<4} {title:<35} {year:<6} {rating:<7} {sim_score:<10}")
        
        print("\n" + "="*80)