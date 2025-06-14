# NIC Code Semantic Search Platform

A comprehensive multilingual semantic search platform for National Industrial Classification (NIC) codes with advanced voice-to-text capabilities, supporting both Hindi and English languages.

## 🌟 Overview

This project provides an intelligent search system that allows users to find relevant NIC codes using natural language queries. The platform features separate services for Hindi and English languages, with advanced voice recognition capabilities, modern web interfaces, and multiple deployment options.

## 🏗️ Architecture

```
semantic-search-nic-deploy/
├── English/                    # English semantic search service
│   ├── static/                # Static web assets
│   ├── templates/             # HTML templates
│   ├── Data Processing/       # Data processing utilities
│   ├── embedding_cache/       # Cached embeddings
│   └── *.py                   # Core Python modules
├── Hindi/                     # Hindi semantic search service
│   ├── static/               # Static web assets
│   ├── templates/            # HTML templates
│   └── *.py                  # Core Python modules
├── templates/                # Main application templates
├── streamlit_app.py          # Streamlit unified interface
└── requirements.txt          # Python dependencies
```

## 🚀 Features

### Core Features
- **Multilingual Support**: Native support for Hindi and English languages
- **Voice-to-Text Search**: Advanced speech recognition with fallback simulation mode
- **Semantic Search**: AI-powered search using sentence transformers and FAISS indexing
- **Multiple Interfaces**: Web applications, REST APIs, and Streamlit dashboard
- **Real-time Audio Processing**: Live audio recording and transcription
- **Robust Error Handling**: Comprehensive error handling with graceful fallbacks

### English Search Service Features
- **Model**: Uses sentence-transformers for semantic embeddings
- **Web Interface**: Modern, responsive Flask-based web application
- **REST API**: FastAPI-based REST endpoints for integration
- **Data Processing**: Advanced text preprocessing and cleaning utilities
- **Caching**: Intelligent embedding caching for improved performance

### Hindi Search Service Features
- **Model**: Specialized Hindi language model (krutrim-ai-labs/Vyakyartha)
- **Voice Recognition**: Hindi speech-to-text using Wav2Vec2 models
- **FAISS Indexing**: Optimized vector search for Hindi content
- **Simulation Mode**: Fallback mode for environments without microphone access
- **Audio Diagnostics**: Comprehensive audio device testing and troubleshooting

## 📋 Prerequisites

### System Requirements
- Python 3.8+ (Python 3.9+ recommended)
- 8GB+ RAM (16GB+ recommended for production)
- GPU support (optional, for accelerated inference)
- Audio input device (for voice features)

### Hardware Requirements
- **Minimum**: 4GB RAM, 2GB storage
- **Recommended**: 8GB RAM, 5GB storage, GPU with 4GB VRAM
- **Production**: 16GB RAM, 10GB storage, GPU with 8GB VRAM

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/semantic-search-nic-deploy.git
cd semantic-search-nic-deploy
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

```

### 4. Download Models
The application will automatically download required models on first run:
- **English**: sentence-transformers models
- **Hindi**: Wav2Vec2 models from Hugging Face
- **Embeddings**: Pre-computed embeddings for NIC codes


```

## 🎯 Usage

### Streamlit Unified Interface

The easiest way to use the platform is through the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Access the application at `http://localhost:8501`

#### Features:
- **Language Selection**: Choose between Hindi and English interfaces
- **Real-time Search**: Type queries and get instant results
- **Voice Input**: Click the microphone button for voice search
- **Results Visualization**: Interactive charts and visualizations
- **Search History**: Track and revisit previous searches

### English Search Service

#### Web Application
```bash
cd English
python app.py
```
Access at `http://localhost:5000`

#### REST API
```bash
cd English
python start_api.py
```
API available at `http://localhost:8000`

##### API Endpoints:
- `POST /search` - Perform semantic search
- `GET /health` - Health check
- `GET /stats` - Index statistics

##### Example API Usage:
```bash
# Search for NIC codes
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "software development", "top_k": 5}'
```

#### Features:
- **Advanced Search**: Natural language processing for accurate results
- **Filtering**: Search by NIC code level, activity type
- **Export**: Download results in JSON/CSV format
- **Performance Metrics**: Search time and relevance scoring

### Hindi Search Service

#### Web Application
```bash
cd Hindi
python hindi_search_webapp.py
```
Access at `http://localhost:5001`

#### Command Line Interface
```bash
cd Hindi
python search_hindi_cli.py "आपका खोज प्रश्न यहां"
```

#### Features:
- **Hindi Voice Search**: Speak in Hindi for voice-to-text search
- **Audio Diagnostics**: Built-in audio device testing
- **Simulation Mode**: Works in environments without microphone
- **Devanagari Support**: Full Unicode support for Hindi text
- **Regional Variations**: Handles different Hindi dialects

## 🎤 Voice-to-Text Features

### Audio Recording
- **Real-time Recording**: Live audio capture with visual feedback
- **Multiple Formats**: Support for various audio formats (WAV, MP3)
- **Automatic Normalization**: Audio level normalization for better recognition
- **Background Recording**: Non-blocking recording with threading

### Speech Recognition
- **Hindi Models**: Vakyansh Wav2Vec2 models for Hindi speech recognition
- **English Models**: Whisper/Wav2Vec2 for English speech recognition
- **Noise Reduction**: Built-in noise filtering and enhancement
- **Multiple Sampling Rates**: Support for different audio quality levels

### Audio Diagnostics
Access audio diagnostics at any time:
```bash
cd Hindi
python recording.py
```

Features:
- **Device Detection**: List all available audio devices
- **Device Testing**: Test each device for compatibility
- **System Information**: Display audio system status
- **Troubleshooting**: Automated problem detection and solutions

## 🔧 Configuration

### Environment Variables
```bash
# Server environment
SERVER_ENV=production|staging|development

# Audio settings
DISABLE_AUDIO=0|1
DETAILED_LOGS=0|1
SIMULATION_MODE=0|1

# Model settings
ENGLISH_MODEL=sentence-transformers/all-MiniLM-L6-v2
HINDI_MODEL=krutrim-ai-labs/Vyakyartha

# Performance settings
MAX_SEARCH_RESULTS=50
CACHE_SIZE=1000
```

### Audio Configuration
```python
# recording.py settings
sample_rate = 16000  # Standard for speech recognition
channels = 1         # Mono audio
chunk_size = 1024    # Audio buffer size
```

## 📊 Performance Optimization

### For English Service:
- **Embedding Caching**: Pre-computed embeddings stored in `embedding_cache/`
- **FAISS Indexing**: Optimized vector search with IVF indexing
- **Batch Processing**: Process multiple queries simultaneously
- **GPU Acceleration**: CUDA support for faster inference

### For Hindi Service:
- **Model Optimization**: Quantized models for faster inference
- **Index Compression**: Compressed FAISS indices for memory efficiency
- **Parallel Processing**: Multi-threaded search and transcription
- **Memory Management**: Efficient memory usage for large datasets

## 🧪 Testing


### Audio Testing
```bash
# Test audio recording
cd Hindi
python recording.py

# Test transcription
python -c "from transcription import transcribe_audio_file; print(transcribe_audio_file('test.wav'))"
```

### API Testing
```bash
# Test English API
curl -X GET "http://localhost:8000/health"

# Test Hindi API
curl -X POST "http://localhost:5001/api/system_info"
```

## 🚀 Deployment

### Production Setup

#### Using Docker (Recommended)
```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Using Docker Compose
```yaml
version: '3.8'
services:
  semantic-search:
    build: .
    ports:
      - "8501:8501"
    environment:
      - SERVER_ENV=production
      - DISABLE_AUDIO=1
    volumes:
      - ./data:/app/data
```

#### Using Gunicorn
```bash
# For English service
cd English
gunicorn --bind 0.0.0.0:8000 -w 4 start_api:app

# For Hindi service
cd Hindi
gunicorn --bind 0.0.0.0:5001 -w 2 hindi_search_webapp:app
```

### Cloud Deployment

#### AWS EC2
```bash
# Install required packages
sudo apt-get update
sudo apt-get install python3-pip python3-venv

# Clone and setup
git clone <repository>
cd semantic-search-nic-deploy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with screen
screen -S semantic-search
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

#### Google Cloud Platform
```bash
# Using Cloud Run
gcloud run deploy semantic-search \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

## 🔒 Security Considerations

### Data Protection
- **Input Validation**: All user inputs are sanitized and validated
- **Rate Limiting**: API endpoints have built-in rate limiting
- **CORS Protection**: Proper CORS headers for web security
- **Audio Privacy**: Audio data is processed locally and not stored

### Authentication (Optional)
```python
# Add authentication middleware
from functools import wraps
from flask import request, jsonify

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not validate_token(auth_header):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated
```

## 🛠️ Development


### Adding New Languages
1. Create language-specific directory (e.g., `Tamil/`)
2. Implement search class following existing patterns
3. Add language configuration to `streamlit_app.py`
4. Update templates and static files
5. Add corresponding models and data files

### Extending Features
- **New Search Algorithms**: Implement in respective language directories
- **Additional Voice Models**: Add to `transcription.py`
- **New UI Components**: Update templates and static files
- **API Endpoints**: Extend FastAPI routes

## 📝 API Documentation

### English Search API

#### Search Endpoint
```http
POST /search
Content-Type: application/json

{
  "query": "software development",
  "top_k": 10,
  "filters": {
    "level": 4,
    "section": "J"
  }
}
```

#### Response Format
```json
{
  "status": "success",
  "results": [
    {
      "nic_code": "62011",
      "description": "Computer programming activities",
      "section": "J",
      "level": 5,
      "similarity_score": 0.89
    }
  ],
  "search_time": 0.045,
  "total_results": 1
}
```

### Hindi Search API

#### Voice Recording Endpoints
```http
POST /api/start_recording
Content-Type: application/json
{}
```

```http
POST /api/stop_recording
Content-Type: application/json
{}
```

#### System Information
```http
GET /api/system_info
```

## 🐛 Troubleshooting

### Common Issues

#### Audio Not Working
1. **Check Permissions**: Ensure browser has microphone access
2. **Test Devices**: Use `python recording.py` to test audio devices
3. **Simulation Mode**: Set `SIMULATION_MODE=1` for testing without microphone
4. **Dependencies**: Ensure `sounddevice` and `librosa` are installed

#### Search Results Empty
1. **Check Models**: Ensure models are downloaded correctly
2. **Index Files**: Verify FAISS index files exist
3. **Data Files**: Check embedding files are present
4. **Memory**: Ensure sufficient RAM for model loading

#### Performance Issues
1. **GPU Setup**: Install GPU versions of PyTorch and FAISS
2. **Memory**: Increase available RAM or reduce batch sizes
3. **Caching**: Enable embedding caching for faster repeated searches
4. **Indexing**: Rebuild FAISS indices with optimal parameters

### Debug Mode
```bash
# Enable detailed logging
export DETAILED_LOGS=1

# Run with debug flags
python -u streamlit_app.py --logger.level=debug
```

### Log Files
```bash
# Check application logs
tail -f logs/application.log

# Check error logs
tail -f logs/error.log
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- **Documentation**: Check this README and API documentation
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the development team

## 🔄 Changelog

### Version 3.0.0
- Unified Streamlit implementation with seamless language switching
- Enhanced user interface with modern design and responsive layout
- Integrated search visualization with interactive charts and graphs
- Advanced result formatting with similarity scores and detailed classifications
- Improved session management and state handling
- Real-time search performance with optimized caching
- Enhanced error handling and user feedback mechanisms

### Version 2.0.0
- Added Hindi language support
- Implemented voice-to-text features
- Added Streamlit unified interface
- Improved error handling and diagnostics

### Version 1.0.0
- Initial release with English semantic search
- REST API implementation
- Basic web interface

## 🙏 Acknowledgments

- **Hugging Face**: For providing excellent pre-trained models
- **FAISS**: For efficient similarity search
- **Streamlit**: For the amazing web framework
- **Vakyansh**: For Hindi speech recognition models
- **Contributors**: All contributors who helped build this project

---

**Happy Searching! 🔍**
