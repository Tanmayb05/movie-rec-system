# 🎬 TMDB Movie Recommendation System

A comprehensive movie recommendation system implementing **10 different content-based filtering algorithms** using the TMDB (The Movie Database) dataset from Kaggle.

## 🚀 Features

### 🔍 Content-Based Filtering Algorithms
1. **TF-IDF with Cosine Similarity** - Text-based recommendations using movie descriptions
2. **Count Vectorizer with Cosine Similarity** - Binary word occurrence similarity
3. **Weighted Multi-Feature Similarity** - Combines genres, cast, keywords, director, and year
4. **Euclidean Distance-Based Similarity** - Numerical feature similarity
5. **Fuzzy String Matching** - Cast and crew name similarity with fuzzy matching
6. **Multi-Feature Categorical Similarity** - Jaccard similarity for categorical features
7. **Clustering-Based Recommendations** - K-means clustering with similarity
8. **Graph-Based Recommendations** - Network analysis of movie relationships
9. **Temporal Content-Based Filtering** - Time-aware recommendations
10. **Hybrid Content-Based Approach** - Combines multiple algorithms

### 📊 Comprehensive Analysis
- **Interactive Visualizations** - Plotly-based interactive charts
- **Algorithm Comparison** - Performance metrics across all algorithms
- **Recommendation Heatmaps** - Visual overlap analysis
- **Genre Distribution Analysis** - Genre patterns in recommendations
- **Detailed Results Tables** - Formatted output with movie details

### 🎯 User-Friendly Interface
- **Interactive Input** - Enter 5 favorite movies
- **Fuzzy Movie Search** - Handles partial movie names
- **Multiple Recommendation Rounds** - Interactive mode for multiple queries
- **Comprehensive Reporting** - Saves results and visualizations

## 📋 Requirements

### System Requirements
- Python 3.7+
- 4GB+ RAM (for processing similarity matrices)
- Internet connection (for dataset download)

### Python Packages
```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
nltk==3.8.1
fuzzywuzzy==0.18.0
python-Levenshtein==0.21.1
kaggle==1.5.16
wordcloud==1.9.2
plotly==5.15.0
sentence-transformers==2.2.2
networkx==3.1
```

## 🛠️ Installation & Setup

### 1. Clone/Download the Project
```bash
# Create project directory
mkdir movie-recommender
cd movie-recommender

# Download all project files:
# - requirements.txt
# - data_loader.py
# - content_algorithms.py
# - visualizer.py
# - main.py
# - setup.py
# - README.md
```

### 2. Automated Setup
```bash
# Run the setup script
python setup.py
```

The setup script will:
- Install all required packages
- Set up Kaggle API credentials
- Create necessary directories
- Download NLTK data
- Test all imports

### 3. Manual Setup (Alternative)

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Setup Kaggle API
1. Go to [Kaggle Account Settings](https://www.kaggle.com/account)
2. Click "Create New API Token"
3. Download `kaggle.json`
4. Place it in `~/.kaggle/kaggle.json` (Unix) or `C:\Users\{username}\.kaggle\kaggle.json` (Windows)

#### Create Directories
```bash
mkdir data visualizations results
```

## 🎮 Usage

### Quick Start
```bash
python main.py
```

### Step-by-Step Process

1. **Enter Your Favorite Movies**
   ```
   Movie 1: The Dark Knight
   Movie 2: Inception
   Movie 3: Interstellar
   Movie 4: Avatar
   Movie 5: The Matrix
   ```

2. **System Processing**
   - Downloads TMDB dataset (first run only)
   - Validates movie titles
   - Runs all 10 algorithms
   - Generates recommendations

3. **View Results**
   - Console output with detailed tables
   - Visualizations saved to `visualizations/` folder
   - Interactive HTML charts

### Sample Output
```
🎬 TMDB Movie Recommendation System
==================================================

📊 Loading and preprocessing data...
✅ System ready! Loaded 4803 movies

🎯 Enter 5 movies you like (one per line):
Movie 1: Inception
Movie 2: The Dark Knight
Movie 3: Interstellar
Movie 4: The Matrix
Movie 5: Avatar

🔍 Searching for movies in database...
✅ Found: 'Inception' → 'Inception'
✅ Found: 'The Dark Knight' → 'The Dark Knight'
✅ Found: 'Interstellar' → 'Interstellar'
✅ Found: 'The Matrix' → 'The Matrix'
✅ Found: 'Avatar' → 'Avatar'

🚀 Running recommendation algorithms...
Based on: Inception, The Dark Knight, Interstellar, The Matrix, Avatar

==================================================
Running Algorithm 1: TF-IDF with Cosine Similarity...
✓ TF-IDF Cosine completed successfully

Running Algorithm 2: Count Vectorizer with Cosine Similarity...
✓ Count Vector Cosine completed successfully

... (continues for all 10 algorithms)
```

## 📁 Project Structure

```
movie-recommender/
├── main.py                 # Main application entry point
├── data_loader.py          # Data loading and preprocessing
├── content_algorithms.py   # All 10 recommendation algorithms
├── visualizer.py          # Visualization and reporting
├── setup.py               # Automated setup script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # TMDB dataset (auto-downloaded)
├── visualizations/       # Generated charts and reports
└── results/              # Saved recommendation results
```

## 🧠 Algorithm Details

### 1. TF-IDF with Cosine Similarity
- **Input**: Movie overviews, genres, keywords
- **Method**: TF-IDF vectorization + cosine similarity
- **Best for**: Content similarity based on descriptions

### 2. Count Vectorizer with Cosine Similarity
- **Input**: Same as TF-IDF but binary occurrence
- **Method**: Count vectorization + cosine similarity
- **Best for**: Genre and keyword matching

### 3. Weighted Multi-Feature Similarity
- **Input**: Genres, cast, keywords, director, year
- **Method**: Weighted combination of multiple similarities
- **Weights**: Genre (35%), Cast (25%), Keywords (20%), Director (15%), Year (5%)

### 4. Euclidean Distance-Based Similarity
- **Input**: Numerical features (budget, revenue, runtime, ratings)
- **Method**: Euclidean distance → similarity conversion
- **Best for**: Movies with similar production values

### 5. Fuzzy String Matching
- **Input**: Cast and crew names
- **Method**: Fuzzy string matching with Levenshtein distance
- **Best for**: Handling name variations and similarities

### 6. Multi-Feature Categorical Similarity
- **Input**: Multiple categorical features
- **Method**: Jaccard similarity for each feature type
- **Best for**: Genre and cast overlap analysis

### 7. Clustering-Based Recommendations
- **Input**: Combined feature matrix
- **Method**: K-means clustering + within-cluster similarity
- **Best for**: Discovering movie groups and patterns

### 8. Graph-Based Recommendations
- **Input**: Movie relationships (shared cast, genres, directors)
- **Method**: NetworkX graph analysis
- **Best for**: Network-based movie connections

### 9. Temporal Content-Based Filtering
- **Input**: Content features + release years
- **Method**: Time-weighted content similarity
- **Best for**: Era-appropriate recommendations

### 10. Hybrid Content-Based Approach
- **Input**: Multiple algorithm outputs
- **Method**: Weighted combination of top algorithms
- **Best for**: Balanced, comprehensive recommendations

## 📊 Visualizations

The system generates comprehensive visualizations:

### 1. Algorithm Performance Comparison
- Bar chart showing average similarity scores per algorithm
- Identifies best-performing algorithms for your taste

### 2. Recommendation Heatmap
- Shows overlap between algorithms
- Identifies consensus recommendations

### 3. Similarity Score Distributions
- Histograms for each algorithm
- Shows score patterns and ranges

### 4. Interactive Comparison (Plotly)
- Interactive scatter plot
- Hover details with movie information
- Algorithm filtering

### 5. Genre Distribution Analysis
- Compares recommended genres to input preferences
- Shows algorithm bias toward certain genres

## 🎯 Use Cases

### Personal Movie Discovery
- Enter movies you love
- Discover similar films across different similarity metrics
- Explore different recommendation approaches

### Algorithm Research
- Compare content-based filtering methods
- Analyze algorithm performance differences
- Study recommendation diversity vs. accuracy

### Movie Analysis
- Understand movie relationships and patterns
- Explore genre and cast connections
- Visualize movie landscape

## 🔧 Customization

### Adding New Algorithms
1. Add method to `ContentBasedRecommender` class
2. Follow naming convention: `algorithm_N_description`
3. Return list of `(movie_index, similarity_score)` tuples
4. Add to `run_all_algorithms` method

### Modifying Weights
Edit weights in `algorithm_3_weighted_features` and `algorithm_10_hybrid_content`:
```python
# Example: Emphasize genre similarity
weighted_sim = (
    0.50 * genre_sim +      # Increased from 0.35
    0.20 * cast_sim +       # Decreased from 0.25
    0.15 * keyword_sim +    # Decreased from 0.20
    0.10 * director_sim +   # Decreased from 0.15
    0.05 * year_sim         # Same
)
```

### Adding New Visualizations
Add methods to `RecommendationVisualizer` class following existing patterns.

## 🐛 Troubleshooting

### Common Issues

#### 1. Kaggle API Error
```
OSError: Could not find kaggle.json
```
**Solution**: Set up Kaggle API credentials properly
```bash
# Check if file exists
ls ~/.kaggle/kaggle.json

# If not, run setup again
python setup.py
```

#### 2. Memory Error
```
MemoryError: Unable to allocate array
```
**Solution**: 
- Close other applications
- Use smaller feature matrices
- Implement batch processing for large datasets

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'package_name'
```
**Solution**:
```bash
pip install -r requirements.txt
```

#### 4. Movie Not Found
```
❌ Not found: 'movie_name'
```
**Solution**:
- Use exact or partial movie titles
- Check year if multiple versions exist
- Use suggestions provided by the system

### Performance Optimization

#### For Large Datasets
- Reduce `max_features` in TF-IDF
- Limit cast/crew to top N
- Use sparse matrices where possible
- Implement caching for similarity matrices

#### For Faster Processing
- Skip computationally expensive algorithms
- Reduce number of recommendations
- Use pre-computed similarity matrices

## 📈 Future Enhancements

### Planned Features
- **Collaborative Filtering** integration
- **Deep Learning** approaches (Neural Collaborative Filtering)
- **Real-time** recommendations
- **Web Interface** with Flask/Django
- **Database Integration** for scalability
- **A/B Testing** framework for algorithms

### Advanced Algorithms
- **Matrix Factorization** techniques
- **Word Embeddings** (Word2Vec, BERT)
- **Ensemble Methods** with voting
- **Reinforcement Learning** for personalization

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Areas for Contribution
- New recommendation algorithms
- Performance optimizations
- Additional visualizations
- Web interface development
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TMDB** for providing the movie dataset
- **Kaggle** for hosting the dataset
- **Scikit-learn** for machine learning tools
- **Plotly** for interactive visualizations
- **NetworkX** for graph analysis capabilities

## 📞 Support

For questions, issues, or suggestions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Review the code documentation
4. Check existing issues for solutions

---

**Happy Movie Discovering! 🍿🎬**