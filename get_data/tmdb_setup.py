#!/usr/bin/env python3
"""
Setup script for TMDB Movie Recommendation System
"""

import os
import subprocess
import sys
import json
from dotenv import load_dotenv

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def setup_kaggle_api():
    """Setup and test Kaggle API credentials from .env file"""
    print("\n🔑 Setting up Kaggle API...")
    
    # Load environment variables
    load_dotenv()
    
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if not username or not key:
        print("❌ Kaggle credentials not found in .env file")
        print("Please add the following to your .env file:")
        print("KAGGLE_USERNAME=your_username")
        print("KAGGLE_KEY=your_api_key")
        print("\nGet your API key from: https://www.kaggle.com/account")
        return False
    
    # Create kaggle directory and credentials file
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")
    
    os.makedirs(kaggle_dir, exist_ok=True)
    
    credentials = {
        "username": username,
        "key": key
    }
    
    with open(kaggle_file, 'w') as f:
        json.dump(credentials, f)
    
    # Set file permissions (Unix-like systems)
    try:
        os.chmod(kaggle_file, 0o600)
    except:
        pass
    
    # Test the API key
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Try to get user info to test authentication
        user_info = api.get_config_value(api.CONFIG_NAME_USER)
        print(f"✅ Kaggle API authenticated successfully for user: {user_info}")
        return True
        
    except Exception as e:
        print(f"❌ Kaggle API authentication failed: {e}")
        print("Please check your credentials in the .env file")
        return False

def setup_tmdb():
    """Setup and test TMDB API key from .env file"""
    print("\n🎬 Setting up TMDB API...")
    
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('TMDB_API_KEY')
    
    if not api_key:
        print("❌ TMDB API key not found in .env file")
        print("Please add the following to your .env file:")
        print("TMDB_API_KEY=your_api_key")
        print("\nGet your API key from: https://www.themoviedb.org/settings/api")
        return False
    
    # Test the API key
    try:
        import tmdbsimple as tmdb
        tmdb.API_KEY = api_key
        
        # Test with a simple API call
        search = tmdb.Search()
        response = search.movie(query='The Matrix')
        
        # Access results from the response dictionary
        results = response.get('results', [])
        if results and len(results) > 0:
            print(f"✅ TMDB API authenticated successfully!")
            print(f"   Test search returned {len(results)} results")
            return True
        else:
            print("❌ TMDB API test failed - no results returned")
            return False
            
    except Exception as e:
        print(f"❌ TMDB API authentication failed: {e}")
        print("Please check your API key in the .env file")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"✅ Created/verified: {data_dir}/")

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded!")
    except Exception as e:
        print(f"⚠️ NLTK download failed: {e}")

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'nltk', 'fuzzywuzzy', 'plotly', 'networkx',
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'dotenv':
                __import__('python_dotenv')
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("\n✅ All packages imported successfully!")
    return True

def main():
    """Main setup function"""
    print("🎬 TMDB Movie Recommendation System Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at package installation")
        return
    
    # Step 2: Test imports
    if not test_imports():
        print("❌ Setup failed at import testing")
        return
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Setup and test TMDB API
    tmdb_success = setup_tmdb()
    
    # Step 5: Setup and test Kaggle API
    kaggle_success = setup_kaggle_api()
    
    # Step 6: Download NLTK data
    download_nltk_data()
    
    print("\n" + "=" * 50)
    
    if tmdb_success and kaggle_success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python tmdb_dataset_collector.py")
        print("2. Then run: python main.py")
        print("3. Enter 5 movies you like")
        print("4. Get personalized recommendations!")
    else:
        print("⚠️ Setup completed with some issues:")
        if not tmdb_success:
            print("   • TMDB API key needs to be configured")
        if not kaggle_success:
            print("   • Kaggle API key needs to be configured")
        print("\nPlease fix the API key issues and run setup again.")

if __name__ == "__main__":
    main()