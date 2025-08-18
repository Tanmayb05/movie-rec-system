#!/usr/bin/env python3
"""
Setup script for TMDB Movie Recommendation System
"""

import os
import subprocess
import sys
import json

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
    """Setup Kaggle API credentials"""
    print("\n🔑 Setting up Kaggle API...")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")
    
    if os.path.exists(kaggle_file):
        print("✅ Kaggle API credentials already exist!")
        return True
    
    print("📝 Kaggle API setup required:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download kaggle.json file")
    print("4. Enter your credentials below:")
    
    username = input("Kaggle Username: ").strip()
    key = input("Kaggle API Key: ").strip()
    
    if not username or not key:
        print("❌ Invalid credentials provided")
        return False
    
    # Create kaggle directory
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Create kaggle.json
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
    
    print("✅ Kaggle API credentials saved!")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["data", "visualizations", "results"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created/verified: {directory}/")

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
        'nltk', 'fuzzywuzzy', 'kaggle', 'plotly', 'networkx'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
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
    
    # Step 3: Setup Kaggle API
    if not setup_kaggle_api():
        print("⚠️ Kaggle API setup incomplete - you may need to set this up manually")
    
    # Step 4: Create directories
    create_directories()
    
    # Step 5: Download NLTK data
    download_nltk_data()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Enter 5 movies you like")
    print("3. Get personalized recommendations!")
    print("\nNote: First run will download the TMDB dataset (~3MB)")

if __name__ == "__main__":
    main()