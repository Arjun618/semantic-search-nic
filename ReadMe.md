# Semantic Search for NIC Codes

![FAISS Powered](https://img.shields.io/badge/Powered%20by-FAISS-blue)
![Multilingual](https://img.shields.io/badge/Languages-English%20%7C%20Hindi-orange)
![Python](https://img.shields.io/badge/Language-Python%203.7+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A high-performance semantic search application for National Industrial Classification (NIC) codes using Facebook AI Similarity Search (FAISS) for vector similarity matching and transformer-based embeddings for natural language understanding.

## 📋 Overview

This application provides semantic (meaning-based) search functionality for National Industrial Classification (NIC) codes in both English and Hindi. Unlike traditional keyword matching, semantic search understands the meaning behind search queries, delivering more relevant and intuitive results for industry classification lookups.

## ✨ Features

- **Semantic Search**: Find relevant results based on meaning, not just keywords
- **Multilingual Support**: Search in both English and Hindi
- **Multiple Search Modes**: 
  - Standard: Balanced precision and recall
  - Strict: High precision, fewer results
  - Relaxed: High recall, more results
- **Voice Search**: Search by voice input (Hindi interface)
- **High Performance**: Fast search using FAISS vector similarity
- **Local Storage**: Uses local JSON files without requiring a database connection
- **Result Visualization**: Visual representation of search results with charts
- **Recent Searches**: Tracks search history for quick reference
- **Responsive Design**: Works on mobile, tablet, and desktop devices
- **Dark/Light Modes**: Customizable UI themes
- **Admin Features**: Tools to manage the search index and view system statistics
- **REST API**: Simple API integration with other applications

## 🔍 How It Works

1. **Text Embeddings**: Industry descriptions are converted to vector embeddings using transformer models
2. **FAISS Index**: Vectors are stored in a FAISS index for efficient similarity search
3. **Search Process**: 
   - User queries are converted to the same vector space
   - FAISS finds the most similar vectors using cosine similarity
   - Results are ranked by relevance and returned to the user

## 🧰 Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python (FastAPI)
- **Vector Search**: Facebook AI Similarity Search (FAISS)
- **Embeddings**: Sentence Transformers
- **Data Storage**: Local JSON files
- **Visualization**: Chart.js
- **Voice Recognition**: Web Speech API

## 📂 Project Structure

```
semantic-search-nic/
├── English/                  # English interface and data
│   ├── Data Processing/      # Data processing scripts
│   ├── static/               # Static assets for English UI
│   ├── templates/            # HTML templates
│   ├── output.json           # English NIC data with embeddings
│   ├── translated_output.json # English data with Hindi translations
│   └── README.md             # English interface documentation
├── Hindi/                    # Hindi interface and data
│   ├── static/               # Static assets for Hindi UI
│   │   ├── css/              # Hindi UI stylesheets
│   │   └── js/               # Hindi UI JavaScript
│   ├── templates/            # HTML templates for Hindi UI
│   └── output_hindi.json     # Hindi NIC data with embeddings
├── static/                   # Shared static assets
│   ├── script.js             # Shared JavaScript
│   └── style.css             # Shared styles
├── templates/                # Shared HTML templates
└── README.md                 # Main project documentation
```

## 🛠️ Installation and Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/semantic-search-nic.git
   cd semantic-search-nic
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### English Interface

```bash
cd English
python app.py
```
Access the English interface at `http://localhost:5000`

#### Hindi Interface

```bash
cd Hindi
python app.py
```
Access the Hindi interface at `http://localhost:5000`

## 🔎 Using the Search

1. **Basic Search**:
   - Type your search query in natural language
   - Example: "software development" or "cultivation of wheat"
   - Press Enter or click the search button

2. **Advanced Search Options**:
   - **Results to show**: Control the number of results (5-50)
   - **Search mode**:
     - Standard: Default balance of relevance
     - Strict: For exact matches only
     - Relaxed: For broader matches
   - **Performance metrics**: Toggle to see search timings

3. **Voice Search** (Hindi interface):
   - Click the microphone icon
   - Speak your query in Hindi
   - The system will automatically search when you finish speaking

4. **Understanding Results**:
   - Results are ranked by similarity score (0-100%)
   - Each result shows the complete NIC hierarchy:
     - Section → Division → Group → Class → Sub-Class

## ⚙️ Admin Features

Access the admin panel by clicking the Admin link in the footer:

1. **Rebuild Index**:
   - Manually rebuild the FAISS index if you've updated your data
   - Useful after adding new NIC codes or modifying existing ones

2. **Clear Embedding Cache**:
   - Clear the in-memory cache of vector embeddings
   - Helps free up memory in long-running instances

3. **Index Statistics**:
   - View details about the current FAISS index
   - Check vector count, dimensions, and other technical metrics

## 🌐 API Documentation

The application includes a REST API for programmatic access:

### Search Endpoint

```
POST /search
```

**Request Body**:
```json
{
  "query": "growing of rice",
  "result_count": 10,
  "search_mode": "standard",
  "show_metrics": true
}
```

**Response**:
```json
{
  "results": [
    {
      "section": "A",
      "division": 1.0,
      "group": 11.0,
      "class": 112.0,
      "subclass": 1121.0,
      "description": "Organic farming of basmati rice",
      "score": 0.952,
      "percentage": 95.2
    },
    ...
  ],
  "count": 10,
  "metrics": {
    "total_time_ms": 152.45,
    "embedding_time_ms": 103.12,
    "index_time_ms": 22.75
  }
}
```

See `README_API.md` for comprehensive API documentation.

## 🔄 Multilingual Support

The application supports both English and Hindi:

### English Interface
- Uses English NIC code descriptions
- English search queries
- Results displayed in English

### Hindi Interface
- Uses Hindi NIC code descriptions
- Supports Hindi search queries
- Voice search in Hindi
- Results displayed in Hindi

## 📊 Performance Considerations

- **First Search**: The first search may take longer (5-10 seconds) as the FAISS index and models are loaded
- **Subsequent Searches**: Typically take 100-300ms
- **Memory Usage**: 
  - Base memory: ~300-500MB
  - Embedding models: ~200-400MB
  - FAISS index: ~50MB-2GB (depends on data size)
- **Optimization**: The application uses caching to improve performance of repeated searches

## 🔧 Troubleshooting

### Common Issues

1. **No Results Found**:
   - Try using more general terms
   - Switch to "Relaxed" search mode
   - Check if you're using industry-specific terminology

2. **Voice Recognition Issues**:
   - Ensure your browser supports the Web Speech API
   - Check microphone permissions
   - Speak clearly and at a moderate pace

3. **Slow Performance**:
   - First search is always slower as models load
   - Ensure you have sufficient RAM
   - Close memory-intensive applications

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Facebook Research for creating FAISS
- Sentence-Transformers team for their excellent embeddings models
- The open-source community for various libraries and tools used in this project