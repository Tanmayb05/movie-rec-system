import tmdbsimple as tmdb
import pandas as pd
import time
import json
import numpy as np
from datetime import datetime
import os

# Set your TMDB API key
tmdb.API_KEY = ''# Replace with your actual API key

class TMDBDatasetCollector:
    def __init__(self, min_vote_count=200, max_movies=15000):
        self.min_vote_count = min_vote_count
        self.max_movies = max_movies
        self.movies_data = []
        self.ratings_data = []
        self.metadata_data = []
        
    def collect_movies_with_200_votes(self):
        """
        Collect all movies with 200+ votes using TMDB discover endpoint
        """
        print(f"🎬 COLLECTING MOVIES WITH {self.min_vote_count}+ VOTES")
        print("=" * 60)
        
        discover = tmdb.Discover()
        page = 1
        total_collected = 0
        
        while total_collected < self.max_movies:
            try:
                print(f"📖 Processing page {page}...")
                
                response = discover.movie(
                    vote_count_gte=self.min_vote_count,
                    sort_by='popularity.desc',
                    page=page
                )
                
                if not discover.results:
                    print("No more results found.")
                    break
                
                for movie in discover.results:
                    if total_collected >= self.max_movies:
                        break
                    
                    self.movies_data.append(movie)
                    total_collected += 1
                
                print(f"   • Collected {len(discover.results)} movies (Total: {total_collected})")
                
                # Check if we've reached the last page
                if page >= discover.total_pages or page >= 500:  # API limit
                    break
                
                page += 1
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error on page {page}: {e}")
                break
        
        print(f"✅ Total movies collected: {len(self.movies_data)}")
        return self.movies_data

    def create_movies_info_dataset(self):
        """
        Dataset 1: Movies Information Dataset
        Core movie details for content-based filtering
        """
        print(f"\n📊 CREATING DATASET 1: MOVIES INFORMATION")
        print("=" * 50)
        
        movies_info = []
        
        for i, movie in enumerate(self.movies_data):
            try:
                movie_id = movie['id']
                
                # Get detailed movie information
                movie_obj = tmdb.Movies(movie_id)
                details = movie_obj.info()
                
                # Extract core information
                movie_info = {
                    'movie_id': movie_id,
                    'title': details.get('title', ''),
                    'original_title': details.get('original_title', ''),
                    'release_date': details.get('release_date', ''),
                    'year': details.get('release_date', '')[:4] if details.get('release_date') else '',
                    'runtime': details.get('runtime', 0),
                    'budget': details.get('budget', 0),
                    'revenue': details.get('revenue', 0),
                    'overview': details.get('overview', ''),
                    'tagline': details.get('tagline', ''),
                    'original_language': details.get('original_language', ''),
                    'adult': details.get('adult', False),
                    'status': details.get('status', ''),
                    
                    # Extract genres
                    'genres': '|'.join([genre['name'] for genre in details.get('genres', [])]),
                    'genre_ids': '|'.join([str(genre['id']) for genre in details.get('genres', [])]),
                    
                    # Extract production companies
                    'production_companies': '|'.join([comp['name'] for comp in details.get('production_companies', [])]),
                    
                    # Extract production countries
                    'production_countries': '|'.join([country['name'] for country in details.get('production_countries', [])]),
                    
                    # Extract spoken languages
                    'spoken_languages': '|'.join([lang['name'] for lang in details.get('spoken_languages', [])])
                }
                
                movies_info.append(movie_info)
                
                if (i + 1) % 100 == 0:
                    print(f"   • Processed {i + 1}/{len(self.movies_data)} movies")
                
                time.sleep(0.05)  # Rate limiting
                
            except Exception as e:
                print(f"   ⚠️  Error processing movie {movie_id}: {e}")
                continue
        
        # Create DataFrame
        movies_df = pd.DataFrame(movies_info)
        movies_df.to_csv('movies_info.csv', index=False)
        print(f"✅ Movies Info Dataset saved: {len(movies_df)} movies")
        print(f"   📁 File: movies_info.csv ({movies_df.shape})")
        
        return movies_df

    def create_movie_ratings_dataset(self):
        """
        Dataset 2: Movie Ratings Summary Dataset
        Aggregated rating information from TMDB
        """
        print(f"\n⭐ CREATING DATASET 2: MOVIE RATINGS SUMMARY")
        print("=" * 50)
        
        ratings_data = []
        
        for i, movie in enumerate(self.movies_data):
            try:
                movie_id = movie['id']
                
                # Get movie details for ratings
                movie_obj = tmdb.Movies(movie_id)
                details = movie_obj.info()
                
                rating_info = {
                    'movie_id': movie_id,
                    'vote_average': details.get('vote_average', 0),
                    'vote_count': details.get('vote_count', 0),
                    'popularity': details.get('popularity', 0),
                    
                    # Additional metrics
                    'imdb_id': details.get('imdb_id', ''),
                    'release_date': details.get('release_date', ''),
                    'title': details.get('title', '')  # For reference
                }
                
                ratings_data.append(rating_info)
                
                if (i + 1) % 200 == 0:
                    print(f"   • Processed {i + 1}/{len(self.movies_data)} movie ratings")
                
                time.sleep(0.02)
                
            except Exception as e:
                print(f"   ⚠️  Error processing ratings for movie {movie_id}: {e}")
                continue
        
        # Create DataFrame
        ratings_df = pd.DataFrame(ratings_data)
        ratings_df.to_csv('movie_ratings.csv', index=False)
        print(f"✅ Movie Ratings Dataset saved: {len(ratings_df)} movies")
        print(f"   📁 File: movie_ratings.csv ({ratings_df.shape})")
        
        return ratings_df

    def create_movie_metadata_dataset(self):
        """
        Dataset 3: Movie Metadata Enhancement Dataset
        Images, videos, external IDs, and additional metadata
        """
        print(f"\n🎭 CREATING DATASET 3: MOVIE METADATA")
        print("=" * 50)
        
        metadata_list = []
        
        for i, movie in enumerate(self.movies_data):
            try:
                movie_id = movie['id']
                movie_obj = tmdb.Movies(movie_id)
                
                # Get basic details
                details = movie_obj.info()
                
                # Get external IDs
                try:
                    external_ids = movie_obj.external_ids()
                    imdb_id = getattr(movie_obj, 'imdb_id', '')
                    facebook_id = getattr(movie_obj, 'facebook_id', '')
                    instagram_id = getattr(movie_obj, 'instagram_id', '')
                    twitter_id = getattr(movie_obj, 'twitter_id', '')
                except:
                    imdb_id = facebook_id = instagram_id = twitter_id = ''
                
                # Get images
                try:
                    images = movie_obj.images()
                    poster_path = details.get('poster_path', '')
                    backdrop_path = details.get('backdrop_path', '')
                except:
                    poster_path = backdrop_path = ''
                
                # Get videos/trailers
                try:
                    videos = movie_obj.videos()
                    video_keys = '|'.join([video['key'] for video in getattr(movie_obj, 'results', [])[:3]])  # Top 3 videos
                except:
                    video_keys = ''
                
                # Get keywords
                try:
                    keywords = movie_obj.keywords()
                    keyword_list = '|'.join([kw['name'] for kw in getattr(movie_obj, 'keywords', [])[:10]])  # Top 10 keywords
                except:
                    keyword_list = ''
                
                metadata_info = {
                    'movie_id': movie_id,
                    'title': details.get('title', ''),  # For reference
                    
                    # External IDs
                    'imdb_id': imdb_id,
                    'facebook_id': facebook_id,
                    'instagram_id': instagram_id,
                    'twitter_id': twitter_id,
                    
                    # Images
                    'poster_path': poster_path,
                    'backdrop_path': backdrop_path,
                    'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else '',
                    'backdrop_url': f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else '',
                    
                    # Videos
                    'video_keys': video_keys,
                    'trailer_url': f"https://www.youtube.com/watch?v={video_keys.split('|')[0]}" if video_keys else '',
                    
                    # Keywords
                    'keywords': keyword_list,
                    
                    # Additional metadata
                    'homepage': details.get('homepage', ''),
                    'belongs_to_collection': details.get('belongs_to_collection', {}).get('name', '') if details.get('belongs_to_collection') else ''
                }
                
                metadata_list.append(metadata_info)
                
                if (i + 1) % 100 == 0:
                    print(f"   • Processed {i + 1}/{len(self.movies_data)} movie metadata")
                
                time.sleep(0.1)  # More delay for multiple API calls
                
            except Exception as e:
                print(f"   ⚠️  Error processing metadata for movie {movie_id}: {e}")
                continue
        
        # Create DataFrame
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.to_csv('movie_metadata.csv', index=False)
        print(f"✅ Movie Metadata Dataset saved: {len(metadata_df)} movies")
        print(f"   📁 File: movie_metadata.csv ({metadata_df.shape})")
        
        return metadata_df

    def create_simulated_user_ratings_dataset(self, num_users=1000, avg_ratings_per_user=50):
        """
        Dataset 4: User Ratings Dataset (Simulated)
        Since TMDB doesn't provide individual user ratings, we'll create realistic simulated data
        """
        print(f"\n👥 CREATING DATASET 4: USER RATINGS (SIMULATED)")
        print("=" * 50)
        
        print(f"   • Simulating {num_users} users")
        print(f"   • Average {avg_ratings_per_user} ratings per user")
        
        user_ratings = []
        movie_ids = [movie['id'] for movie in self.movies_data]
        
        # Get movie popularities and ratings for realistic simulation
        movie_weights = {}
        for movie in self.movies_data:
            movie_id = movie['id']
            popularity = movie.get('popularity', 1)
            vote_average = movie.get('vote_average', 5)
            vote_count = movie.get('vote_count', 200)
            
            # Weight by popularity and vote count for realistic distribution
            weight = (popularity * np.log(vote_count + 1)) / 100
            movie_weights[movie_id] = weight
        
        # Normalize weights
        total_weight = sum(movie_weights.values())
        for movie_id in movie_weights:
            movie_weights[movie_id] = movie_weights[movie_id] / total_weight
        
        for user_id in range(1, num_users + 1):
            # Simulate user preferences
            user_genre_preference = np.random.choice(['action', 'comedy', 'drama', 'horror', 'romance', 'sci-fi', 'general'])
            user_rating_bias = np.random.normal(0, 0.5)  # Some users rate higher/lower on average
            
            # Number of movies this user has rated
            num_ratings = max(10, int(np.random.poisson(avg_ratings_per_user)))
            
            # Select movies based on popularity (more popular movies get more ratings)
            movie_ids_list = list(movie_weights.keys())
            weights_list = list(movie_weights.values())
            
            selected_movies = np.random.choice(
                movie_ids_list,
                size=min(num_ratings, len(movie_ids_list)),
                replace=False,
                p=weights_list
            )
            
            for movie_id in selected_movies:
                # Find the actual movie data
                movie_data = next((m for m in self.movies_data if m['id'] == movie_id), None)
                if not movie_data:
                    continue
                
                # Base rating from TMDB average
                tmdb_avg = movie_data.get('vote_average', 5)
                
                # Simulate realistic rating with some noise
                base_rating = tmdb_avg + user_rating_bias
                noise = np.random.normal(0, 1)
                simulated_rating = base_rating + noise
                
                # Clamp to valid range and round to 0.5
                rating = max(1, min(10, round(simulated_rating * 2) / 2))
                
                # Simulate timestamp (random time in last 2 years)
                timestamp = int(time.time()) - np.random.randint(0, 2 * 365 * 24 * 3600)
                
                user_ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    'date_rated': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                })
            
            if user_id % 100 == 0:
                print(f"   • Generated ratings for {user_id}/{num_users} users")
        
        # Create DataFrame
        user_ratings_df = pd.DataFrame(user_ratings)
        user_ratings_df.to_csv('user_ratings.csv', index=False)
        
        print(f"✅ User Ratings Dataset saved: {len(user_ratings_df)} ratings")
        print(f"   📁 File: user_ratings.csv ({user_ratings_df.shape})")
        print(f"   📊 {num_users} users, avg {len(user_ratings_df)/num_users:.1f} ratings per user")
        
        return user_ratings_df

    def generate_summary_report(self):
        """
        Generate a summary report of all collected datasets
        """
        print(f"\n📋 DATASET COLLECTION SUMMARY REPORT")
        print("=" * 60)
        
        datasets = [
            ('movies_info.csv', 'Movies Information'),
            ('movie_ratings.csv', 'Movie Ratings Summary'),
            ('movie_metadata.csv', 'Movie Metadata'),
            ('user_ratings.csv', 'User Ratings (Simulated)')
        ]
        
        total_size = 0
        
        for filename, description in datasets:
            if os.path.exists(filename):
                size_mb = os.path.getsize(filename) / (1024 * 1024)
                total_size += size_mb
                
                # Load and get basic stats
                df = pd.read_csv(filename)
                
                print(f"\n📁 {description}:")
                print(f"   • File: {filename}")
                print(f"   • Size: {size_mb:.2f} MB")
                print(f"   • Rows: {len(df):,}")
                print(f"   • Columns: {len(df.columns)}")
                print(f"   • Sample columns: {', '.join(df.columns[:5])}")
        
        print(f"\n🎯 TOTAL DATASET SIZE: {total_size:.2f} MB")
        print(f"🎬 MOVIES WITH 200+ VOTES: {len(self.movies_data):,}")
        
        print(f"\n💡 NEXT STEPS:")
        print("1. Load datasets using pandas: pd.read_csv('filename.csv')")
        print("2. Explore data with df.head(), df.info(), df.describe()")
        print("3. Start building content-based recommender using movies_info.csv")
        print("4. Build collaborative filtering using user_ratings.csv")
        print("5. Combine both approaches for hybrid recommendations")

def main():
    """
    Main function to collect all 4 core datasets
    """
    print("🎭 TMDB 4 CORE DATASETS COLLECTOR")
    print("=" * 50)
    print("Target: Movies with 200+ votes")
    print("Output: 4 CSV files for movie recommender system")
    print()
    
    if tmdb.API_KEY == 'YOUR_API_KEY_HERE':
        print("❌ Please set your TMDB API key in the script!")
        print("Get your API key from: https://www.themoviedb.org/settings/api")
        return
    
    # Initialize collector
    collector = TMDBDatasetCollector(min_vote_count=200, max_movies=12000)
    
    try:
        # Step 1: Collect movies with 200+ votes
        movies = collector.collect_movies_with_200_votes()
        
        if not movies:
            print("❌ No movies collected. Check your API key and connection.")
            return
        
        # Step 2: Create Dataset 1 - Movies Information
        movies_info_df = collector.create_movies_info_dataset()
        
        # Step 3: Create Dataset 2 - Movie Ratings Summary
        ratings_df = collector.create_movie_ratings_dataset()
        
        # Step 4: Create Dataset 3 - Movie Metadata
        metadata_df = collector.create_movie_metadata_dataset()
        
        # Step 5: Create Dataset 4 - User Ratings (Simulated)
        user_ratings_df = collector.create_simulated_user_ratings_dataset(
            num_users=1000, 
            avg_ratings_per_user=60
        )
        
        # Step 6: Generate summary report
        collector.generate_summary_report()
        
        print(f"\n🎉 SUCCESS! All 4 datasets collected successfully!")
        print("Ready to build your movie recommender system!")
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        print("Check your internet connection and API key.")

if __name__ == "__main__":
    main()