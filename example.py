#!/usr/bin/env python3
"""
Example run script for TMDB Movie Recommendation System
This demonstrates how to use the system programmatically
"""

import sys
import warnings
import time
import logging
warnings.filterwarnings('ignore')

# Load environment variables at the start
from dotenv import load_dotenv
load_dotenv()

from data_loader import TMDBDataLoader
from content_algorithms import ContentBasedRecommender
from visualizer import RecommendationVisualizer

# Setup logging for the example
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_recommendation():
    """Example of running recommendations programmatically"""
    
    print("🎬 TMDB Movie Recommendation System - Example Run")
    print("=" * 60)
    
    # 1. Initialize system
    print("\n📊 Initializing system...")
    start_time = time.time()
    
    loader = TMDBDataLoader()
    df = loader.preprocess_data()
    
    if df is None or df.empty:
        print("❌ Failed to load data")
        return
    
    init_time = time.time() - start_time
    print(f"✅ System initialized in {init_time:.2f}s - Loaded {len(df)} movies")
    
    # 2. Example input movies
    example_movies = [
        "The Dark Knight",
        "Inception", 
        "Interstellar",
        "The Matrix",
        "Avatar"
    ]
    
    print(f"\n🎯 Example input movies: {', '.join(example_movies)}")
    
    # 3. Find movie indices
    search_start_time = time.time()
    movie_indices = []
    found_movies = []
    
    for movie in example_movies:
        idx = loader.find_movie_index(movie)
        if idx >= 0:
            movie_indices.append(idx)
            found_movies.append(df.iloc[idx]['title'])
            print(f"✅ Found: {movie} → {df.iloc[idx]['title']}")
        else:
            print(f"❌ Not found: {movie}")
    
    search_time = time.time() - search_start_time
    print(f"🔍 Movie search completed in {search_time:.2f}s")
    
    if not movie_indices:
        print("❌ No movies found!")
        return
    
    # 4. Initialize recommender and visualizer
    print(f"\n🚀 Initializing recommendation engines...")
    recommender = ContentBasedRecommender(df)
    visualizer = RecommendationVisualizer(df)
    
    # 5. Run all algorithms with timing
    print(f"\n🎯 Running all recommendation algorithms...")
    print(f"Input movies: {', '.join(found_movies)}")
    
    algo_start_time = time.time()
    results = recommender.run_all_algorithms(movie_indices, n_recommendations=10)
    algo_total_time = time.time() - algo_start_time
    
    # 6. Display results
    print(f"\n📊 Displaying results...")
    visualizer.print_detailed_results(results, found_movies)
    
    # 7. Generate visualizations
    print(f"\n📈 Generating visualizations...")
    viz_start_time = time.time()
    visualizer.create_comprehensive_report(results, found_movies, "example_visualizations")
    viz_time = time.time() - viz_start_time
    print(f"✅ Visualizations generated in {viz_time:.2f}s")
    
    # 8. Enhanced Analysis summary
    print(f"\n📋 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Algorithm performance analysis
    algorithm_scores = {}
    algorithm_counts = {}
    
    for algo_name, recommendations in results.items():
        if recommendations:
            scores = [sim for _, sim in recommendations]
            algorithm_scores[algo_name] = {
                'avg_score': sum(scores) / len(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'count': len(recommendations)
            }
            algorithm_counts[algo_name] = len(recommendations)
            
            print(f"{algo_name:<25}: Avg={algorithm_scores[algo_name]['avg_score']:.4f}, "
                  f"Max={algorithm_scores[algo_name]['max_score']:.4f}, "
                  f"Count={algorithm_scores[algo_name]['count']}")
        else:
            print(f"{algo_name:<25}: No results")
    
    # Best performing algorithm
    if algorithm_scores:
        best_algo = max(algorithm_scores.items(), key=lambda x: x[1]['avg_score'])
        most_diverse = max(algorithm_scores.items(), key=lambda x: x[1]['max_score'] - x[1]['min_score'])
        
        print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
        print(f"Best Average Score: {best_algo[0]} ({best_algo[1]['avg_score']:.4f})")
        print(f"Most Diverse Results: {most_diverse[0]} (range: {most_diverse[1]['max_score'] - most_diverse[1]['min_score']:.4f})")
    
    # Top consensus recommendations
    print(f"\n🎯 TOP CONSENSUS RECOMMENDATIONS:")
    print("-" * 60)
    
    # Count how many algorithms recommend each movie
    movie_counts = {}
    for algo_name, recommendations in results.items():
        for movie_idx, sim in recommendations[:5]:  # Top 5 from each
            movie_title = df.iloc[movie_idx]['title']
            if movie_title not in movie_counts:
                movie_counts[movie_title] = {
                    'count': 0, 
                    'total_sim': 0, 
                    'algorithms': [],
                    'year': df.iloc[movie_idx].get('release_year', 'N/A'),
                    'rating': df.iloc[movie_idx].get('vote_average', 'N/A'),
                    'genres': df.iloc[movie_idx].get('genres_str', 'N/A')
                }
            movie_counts[movie_title]['count'] += 1
            movie_counts[movie_title]['total_sim'] += sim
            movie_counts[movie_title]['algorithms'].append(algo_name)
    
    # Sort by count, then by average similarity
    consensus_movies = []
    for movie, data in movie_counts.items():
        avg_sim = data['total_sim'] / data['count']
        consensus_movies.append((movie, data['count'], avg_sim, data['year'], data['rating'], data['genres']))
    
    consensus_movies.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print(f"{'Rank':<4} {'Movie':<30} {'Algorithms':<4} {'Avg Sim':<8} {'Year':<6} {'Rating':<6} {'Genres':<20}")
    print("-" * 90)
    
    for i, (movie, count, avg_sim, year, rating, genres) in enumerate(consensus_movies[:15], 1):
        movie_short = movie[:27] + "..." if len(movie) > 30 else movie
        year_str = str(int(year)) if isinstance(year, (int, float)) and not pd.isna(year) else 'N/A'
        rating_str = f"{rating:.1f}" if isinstance(rating, (int, float)) and not pd.isna(rating) else 'N/A'
        genres_short = genres[:17] + "..." if len(str(genres)) > 20 else str(genres)[:20]
        
        print(f"{i:3d}. {movie_short:<30} {count:2d}/10  {avg_sim:7.3f}  {year_str:<6} {rating_str:<6} {genres_short:<20}")
    
    # Genre analysis of recommendations
    print(f"\n📊 GENRE ANALYSIS:")
    print("-" * 40)
    
    input_genres = set()
    for idx in movie_indices:
        input_genres.update(df.iloc[idx]['genres_list'])
    
    recommended_genres = {}
    for algo_name, recommendations in results.items():
        for movie_idx, _ in recommendations[:10]:
            for genre in df.iloc[movie_idx]['genres_list']:
                if genre not in recommended_genres:
                    recommended_genres[genre] = {'count': 0, 'in_input': genre in input_genres}
                recommended_genres[genre]['count'] += 1
    
    # Sort by count
    sorted_genres = sorted(recommended_genres.items(), key=lambda x: x[1]['count'], reverse=True)
    
    print(f"Input genres: {', '.join(sorted(input_genres))}")
    print(f"\nTop recommended genres:")
    for genre, data in sorted_genres[:10]:
        marker = "★" if data['in_input'] else " "
        print(f"{marker} {genre:<20}: {data['count']:3d} recommendations")
    
    # System performance summary
    print(f"\n⚡ SYSTEM PERFORMANCE SUMMARY:")
    print("-" * 40)
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s")
    print(f"  - System initialization: {init_time:.2f}s")
    print(f"  - Movie search: {search_time:.2f}s") 
    print(f"  - Algorithm execution: {algo_total_time:.2f}s")
    print(f"  - Visualization generation: {viz_time:.2f}s")
    print(f"Recommendations per second: {sum(len(recs) for recs in results.values()) / algo_total_time:.1f}")
    
    print(f"\n✅ Example run completed successfully!")
    print(f"📁 Visualizations saved to 'example_visualizations/' folder")
    print(f"📊 Check the interactive HTML report for detailed analysis")

def test_individual_algorithms():
    """Test individual algorithms separately with timing"""
    
    print("\n🧪 TESTING INDIVIDUAL ALGORITHMS")
    print("=" * 50)
    
    # Load data
    start_time = time.time()
    loader = TMDBDataLoader()
    df = loader.preprocess_data()
    
    if df is None or df.empty:
        print("❌ Failed to load data")
        return
        
    recommender = ContentBasedRecommender(df)
    setup_time = time.time() - start_time
    
    print(f"✅ Setup completed in {setup_time:.2f}s")
    
    # Test movie
    test_movie = "The Dark Knight"
    movie_idx = loader.find_movie_index(test_movie)
    
    if movie_idx < 0:
        print(f"❌ Test movie '{test_movie}' not found")
        return
    
    print(f"🎯 Testing with: {df.iloc[movie_idx]['title']}")
    
    # Test each algorithm individually with detailed timing
    algorithms = [
        ("TF-IDF Cosine", recommender.algorithm_1_tfidf_cosine),
        ("Count Vector", recommender.algorithm_2_count_vectorizer_cosine),
        ("Weighted Features", recommender.algorithm_3_weighted_features),
        ("Euclidean", recommender.algorithm_4_euclidean_similarity),
        ("Fuzzy Cast", recommender.algorithm_5_fuzzy_cast_similarity),
        ("Categorical", recommender.algorithm_6_categorical_similarity),
        ("Clustering", recommender.algorithm_7_clustering_based),
        ("Graph-Based", recommender.algorithm_8_graph_based),
        ("Temporal", recommender.algorithm_9_temporal_content),
        ("Hybrid", recommender.algorithm_10_hybrid_content)
    ]
    
    timing_results = []
    
    for name, algorithm in algorithms:
        try:
            print(f"\n🔍 Testing {name}...")
            start_time = time.time()
            recommendations = algorithm([movie_idx], n_recommendations=5)
            exec_time = time.time() - start_time
            
            timing_results.append((name, exec_time, len(recommendations)))
            
            if recommendations:
                print(f"✅ {name}: {len(recommendations)} recommendations in {exec_time:.3f}s")
                # Show top 3
                for i, (idx, sim) in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {df.iloc[idx]['title']} (similarity: {sim:.3f})")
            else:
                print(f"❌ {name}: No recommendations in {exec_time:.3f}s")
                
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")
            timing_results.append((name, 0, 0))
    
    # Performance summary
    print(f"\n⚡ INDIVIDUAL ALGORITHM PERFORMANCE:")
    print("-" * 50)
    print(f"{'Algorithm':<20} {'Time (s)':<10} {'Recommendations':<15} {'Recs/sec':<10}")
    print("-" * 50)
    
    for name, exec_time, rec_count in timing_results:
        recs_per_sec = rec_count / exec_time if exec_time > 0 else 0
        print(f"{name:<20} {exec_time:8.3f}  {rec_count:13d}   {recs_per_sec:8.1f}")

def performance_benchmark():
    """Comprehensive performance benchmark with multiple test cases"""
    
    print("\n⏱️ COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    import time
    
    # Load data
    start_time = time.time()
    loader = TMDBDataLoader()
    df = loader.preprocess_data()
    
    if df is None or df.empty:
        print("❌ Failed to load data")
        return
        
    recommender = ContentBasedRecommender(df)
    setup_time = time.time() - start_time
    
    print(f"🎯 Setup completed in {setup_time:.2f}s")
    print(f"📊 Dataset size: {len(df)} movies")
    
    # Multiple test scenarios
    test_scenarios = [
        {
            'name': 'Single Movie Test',
            'movies': ["Avatar"],
            'recs': 10
        },
        {
            'name': 'Multiple Movies Test', 
            'movies': ["Avatar", "Inception", "The Matrix"],
            'recs': 10
        },
        {
            'name': 'Large Recommendation Set',
            'movies': ["Avatar", "Inception"],
            'recs': 50
        }
    ]
    
    algorithms = [
        ("TF-IDF", recommender.algorithm_1_tfidf_cosine),
        ("Count Vector", recommender.algorithm_2_count_vectorizer_cosine),
        ("Weighted Features", recommender.algorithm_3_weighted_features),
        ("Euclidean", recommender.algorithm_4_euclidean_similarity),
        ("Fuzzy Cast", recommender.algorithm_5_fuzzy_cast_similarity),
        ("Categorical", recommender.algorithm_6_categorical_similarity),
        ("Clustering", recommender.algorithm_7_clustering_based),
        ("Graph-Based", recommender.algorithm_8_graph_based),
        ("Temporal", recommender.algorithm_9_temporal_content),
        ("Hybrid", recommender.algorithm_10_hybrid_content)
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\n🎬 {scenario['name']}")
        print("-" * 40)
        
        # Find movie indices
        movie_indices = []
        for movie in scenario['movies']:
            idx = loader.find_movie_index(movie)
            if idx >= 0:
                movie_indices.append(idx)
        
        if not movie_indices:
            print(f"❌ No movies found for {scenario['name']}")
            continue
        
        print(f"Movies: {', '.join(scenario['movies'])}")
        print(f"Recommendations requested: {scenario['recs']}")
        
        scenario_results = []
        
        for name, algorithm in algorithms:
            try:
                start_time = time.time()
                recommendations = algorithm(movie_indices, n_recommendations=scenario['recs'])
                exec_time = time.time() - start_time
                
                scenario_results.append({
                    'algorithm': name,
                    'time': exec_time,
                    'recommendations': len(recommendations),
                    'recs_per_sec': len(recommendations) / exec_time if exec_time > 0 else 0
                })
                
                print(f"{name:<20}: {exec_time:6.3f}s ({len(recommendations)} recs, {len(recommendations)/exec_time:5.1f} recs/s)")
                
            except Exception as e:
                print(f"{name:<20}: ERROR - {str(e)}")
                scenario_results.append({
                    'algorithm': name,
                    'time': 0,
                    'recommendations': 0,
                    'recs_per_sec': 0
                })
        
        all_results[scenario['name']] = scenario_results
    
    # Overall benchmark summary
    print(f"\n📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Calculate averages across scenarios
    algorithm_averages = {}
    for scenario_name, results in all_results.items():
        for result in results:
            algo = result['algorithm']
            if algo not in algorithm_averages:
                algorithm_averages[algo] = {'times': [], 'recs_per_sec': []}
            
            if result['time'] > 0:  # Only include successful runs
                algorithm_averages[algo]['times'].append(result['time'])
                algorithm_averages[algo]['recs_per_sec'].append(result['recs_per_sec'])
    
    print(f"{'Algorithm':<20} {'Avg Time (s)':<12} {'Avg Recs/sec':<12} {'Reliability':<12}")
    print("-" * 60)
    
    for algo, data in algorithm_averages.items():
        avg_time = sum(data['times']) / len(data['times']) if data['times'] else 0
        avg_rps = sum(data['recs_per_sec']) / len(data['recs_per_sec']) if data['recs_per_sec'] else 0
        reliability = len(data['times']) / len(test_scenarios) * 100  # Success rate
        
        print(f"{algo:<20} {avg_time:10.3f}  {avg_rps:10.1f}  {reliability:9.1f}%")
    
    # Performance recommendations
    if algorithm_averages:
        fastest_algo = min(algorithm_averages.items(), 
                          key=lambda x: sum(x[1]['times'])/len(x[1]['times']) if x[1]['times'] else float('inf'))
        
        most_efficient = max(algorithm_averages.items(),
                           key=lambda x: sum(x[1]['recs_per_sec'])/len(x[1]['recs_per_sec']) if x[1]['recs_per_sec'] else 0)
        
        print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
        print(f"Fastest Algorithm: {fastest_algo[0]}")
        print(f"Most Efficient: {most_efficient[0]} (recs/second)")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("- For real-time applications: Use TF-IDF or Count Vector")
        print("- For best quality: Use Weighted Features or Hybrid")
        print("- For large datasets: Avoid Fuzzy Cast and Graph-Based")

def main():
    """Main function with enhanced options"""
    
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
        
        if option == "test":
            test_individual_algorithms()
        elif option == "benchmark":
            performance_benchmark()
        elif option == "example":
            example_recommendation()
        elif option == "all":
            print("🚀 Running comprehensive test suite...")
            example_recommendation()
            test_individual_algorithms() 
            performance_benchmark()
        else:
            print("❌ Invalid option. Use: example, test, benchmark, or all")
            print("Examples:")
            print("  python example_run.py example    # Run example recommendation")
            print("  python example_run.py test       # Test individual algorithms")
            print("  python example_run.py benchmark  # Performance benchmark")
            print("  python example_run.py all        # Run everything")
    else:
        # Default: run example
        example_recommendation()

if __name__ == "__main__":
    # Import pandas here to avoid issues
    import pandas as pd
    main()
    """Example of running recommendations programmatically"""
    
    print("🎬 TMDB Movie Recommendation System - Example Run")
    print("=" * 60)
    
    # 1. Initialize system
    print("\n📊 Initializing system...")
    loader = TMDBDataLoader()
    df = loader.preprocess_data()
    
    if df is None or df.empty:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(df)} movies")
    
    # 2. Example input movies
    example_movies = [
        "The Dark Knight",
        "Inception", 
        "Interstellar",
        "The Matrix",
        "Avatar"
    ]
    
    print(f"\n🎯 Example input movies: {', '.join(example_movies)}")
    
    # 3. Find movie indices
    movie_indices = []
    found_movies = []
    
    for movie in example_movies:
        idx = loader.find_movie_index(movie)
        if idx >= 0:
            movie_indices.append(idx)
            found_movies.append(df.iloc[idx]['title'])
            print(f"✅ Found: {movie} → {df.iloc[idx]['title']}")
        else:
            print(f"❌ Not found: {movie}")
    
    if not movie_indices:
        print("❌ No movies found!")
        return
    
    # 4. Initialize recommender and visualizer
    recommender = ContentBasedRecommender(df)
    visualizer = RecommendationVisualizer(df)
    
    # 5. Run all algorithms
    print(f"\n🚀 Running all recommendation algorithms...")
    results = recommender.run_all_algorithms(movie_indices, n_recommendations=10)
    
    # 6. Display results
    print(f"\n📊 Displaying results...")
    visualizer.print_detailed_results(results, found_movies)
    
    # 7. Generate visualizations
    print(f"\n📈 Generating visualizations...")
    visualizer.create_comprehensive_report(results, found_movies, "example_visualizations")
    
    # 8. Analysis summary
    print(f"\n📋 ANALYSIS SUMMARY")
    print("=" * 40)
    
    algorithm_scores = {}
    for algo_name, recommendations in results.items():
        if recommendations:
            avg_score = sum(sim for _, sim in recommendations) / len(recommendations)
            algorithm_scores[algo_name] = avg_score
            print(f"{algo_name:<25}: {avg_score:.4f}")
        else:
            print(f"{algo_name:<25}: No results")
    
    # Best performing algorithm
    if algorithm_scores:
        best_algo = max(algorithm_scores.items(), key=lambda x: x[1])
        print(f"\n🏆 Best performing algorithm: {best_algo[0]} (Score: {best_algo[1]:.4f})")
    
    # Top consensus recommendations
    print(f"\n🎯 TOP CONSENSUS RECOMMENDATIONS:")
    print("-" * 40)
    
    # Count how many algorithms recommend each movie
    movie_counts = {}
    for algo_name, recommendations in results.items():
        for movie_idx, sim in recommendations[:5]:  # Top 5 from each
            movie_title = df.iloc[movie_idx]['title']
            if movie_title not in movie_counts:
                movie_counts[movie_title] = {'count': 0, 'total_sim': 0}
            movie_counts[movie_title]['count'] += 1
            movie_counts[movie_title]['total_sim'] += sim
    
    # Sort by count, then by average similarity
    consensus_movies = []
    for movie, data in movie_counts.items():
        avg_sim = data['total_sim'] / data['count']
        consensus_movies.append((movie, data['count'], avg_sim))
    
    consensus_movies.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    for i, (movie, count, avg_sim) in enumerate(consensus_movies[:10], 1):
        print(f"{i:2d}. {movie:<35} ({count}/10 algorithms, avg: {avg_sim:.3f})")
    
    print(f"\n✅ Example run completed!")
    print(f"📁 Visualizations saved to 'example_visualizations/' folder")

def test_individual_algorithms():
    """Test individual algorithms separately"""
    
    print("\n🧪 TESTING INDIVIDUAL ALGORITHMS")
    print("=" * 50)
    
    # Load data
    loader = TMDBDataLoader()
    df = loader.preprocess_data()
    recommender = ContentBasedRecommender(df)
    
    # Test movie
    test_movie = "The Dark Knight"
    movie_idx = loader.find_movie_index(test_movie)
    
    if movie_idx < 0:
        print(f"❌ Test movie '{test_movie}' not found")
        return
    
    print(f"🎯 Testing with: {df.iloc[movie_idx]['title']}")
    
    # Test each algorithm individually
    algorithms = [
        ("TF-IDF Cosine", recommender.algorithm_1_tfidf_cosine),
        ("Count Vector", recommender.algorithm_2_count_vectorizer_cosine),
        ("Weighted Features", recommender.algorithm_3_weighted_features),
        ("Euclidean", recommender.algorithm_4_euclidean_similarity),
        ("Fuzzy Cast", recommender.algorithm_5_fuzzy_cast_similarity),
        ("Categorical", recommender.algorithm_6_categorical_similarity),
        ("Clustering", recommender.algorithm_7_clustering_based),
        ("Graph-Based", recommender.algorithm_8_graph_based),
        ("Temporal", recommender.algorithm_9_temporal_content),
        ("Hybrid", recommender.algorithm_10_hybrid_content)
    ]
    
    for name, algorithm in algorithms:
        try:
            print(f"\n🔍 Testing {name}...")
            recommendations = algorithm([movie_idx], n_recommendations=5)
            
            if recommendations:
                print(f"✅ {name}: {len(recommendations)} recommendations")
                # Show top 3
                for i, (idx, sim) in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {df.iloc[idx]['title']} (sim: {sim:.3f})")
            else:
                print(f"❌ {name}: No recommendations")
                
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")

def performance_benchmark():
    """Benchmark algorithm performance"""
    
    print("\n⏱️ PERFORMANCE BENCHMARK")
    print("=" * 40)
    
    import time
    
    # Load data
    loader = TMDBDataLoader()
    df = loader.preprocess_data()
    recommender = ContentBasedRecommender(df)
    
    # Test movies
    test_movies = ["Avatar", "Inception", "The Matrix"]
    movie_indices = []
    
    for movie in test_movies:
        idx = loader.find_movie_index(movie)
        if idx >= 0:
            movie_indices.append(idx)
    
    if not movie_indices:
        print("❌ No test movies found")
        return
    
    print(f"🎯 Benchmarking with {len(movie_indices)} movies")
    
    # Benchmark each algorithm
    algorithms = [
        ("TF-IDF", recommender.algorithm_1_tfidf_cosine),
        ("Count Vector", recommender.algorithm_2_count_vectorizer_cosine),
        ("Weighted Features", recommender.algorithm_3_weighted_features),
        ("Euclidean", recommender.algorithm_4_euclidean_similarity),
        ("Fuzzy Cast", recommender.algorithm_5_fuzzy_cast_similarity),
        ("Categorical", recommender.algorithm_6_categorical_similarity),
        ("Clustering", recommender.algorithm_7_clustering_based),
        ("Graph-Based", recommender.algorithm_8_graph_based),
        ("Temporal", recommender.algorithm_9_temporal_content),
        ("Hybrid", recommender.algorithm_10_hybrid_content)
    ]
    
    benchmark_results = []
    
    for name, algorithm in algorithms:
        try:
            start_time = time.time()
            recommendations = algorithm(movie_indices, n_recommendations=10)
            end_time = time.time()
            
            execution_time = end_time - start_time
            benchmark_results.append((name, execution_time, len(recommendations)))
            
            print(f"{name:<20}: {execution_time:.3f}s ({len(recommendations)} recommendations)")
            
        except Exception as e:
            print(f"{name:<20}: ERROR - {str(e)}")
    
    # Summary
    if benchmark_results:
        fastest = min(benchmark_results, key=lambda x: x[1])
        slowest = max(benchmark_results, key=lambda x: x[1])
        avg_time = sum(x[1] for x in benchmark_results) / len(benchmark_results)
        
        print(f"\n📊 BENCHMARK SUMMARY:")
        print(f"Fastest: {fastest[0]} ({fastest[1]:.3f}s)")
        print(f"Slowest: {slowest[0]} ({slowest[1]:.3f}s)")
        print(f"Average: {avg_time:.3f}s")

def main():
    """Main function with options"""
    
    if len(sys.argv) > 1:
        option = sys.argv[1]
        
        if option == "test":
            test_individual_algorithms()
        elif option == "benchmark":
            performance_benchmark()
        elif option == "example":
            example_recommendation()
        else:
            print("Invalid option. Use: test, benchmark, or example")
    else:
        # Default: run example
        example_recommendation()

if __name__ == "__main__":
    main()