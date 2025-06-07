# Semantic Search Portal

A unified portal for accessing both English and Hindi semantic search services. This main application automatically starts and manages both language-specific search engines, providing a single entry point for users.

## Features

- 🌐 **Unified Portal**: Single entry point for both English and Hindi semantic search
- 🚀 **Auto-Service Management**: Automatically starts both search services
- 📊 **Real-time Status**: Live monitoring of service health and availability
- 🎨 **Modern UI**: Beautiful, responsive interface with language selection
- 🔄 **Auto-Recovery**: Automatic retry mechanisms for failed services
- 📱 **Mobile Friendly**: Responsive design that works on all devices

## Quick Start

### Option 1: Using the Startup Script (Recommended)
```bash
./start_portal.sh
```

### Option 2: Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Start the portal
python3 main_app.py
```

## Access Points

Once started, you can access:

- **Main Portal**: http://localhost:3000 (Language selection page)
- **English Search**: http://localhost:5000 (Direct access)
- **Hindi Search**: http://localhost:5500 (Direct access)

## Architecture

```
Main Portal (Port 3000)
├── English Semantic Search Service (Port 5000)
│   ├── FAISS Index for English documents
│   ├── Sentence Transformers model
│   └── MongoDB/JSON data source
└── Hindi Semantic Search Service (Port 5500)
    ├── FAISS Index for Hindi documents
    ├── Krutrim Hindi embedding model
    └── Hindi document corpus
```

## Service Management

The main application provides several management features:

### API Endpoints

- `GET /` - Main language selection page
- `GET /status` - JSON status of all services
- `GET /english` - Redirect to English search (if available)
- `GET /hindi` - Redirect to Hindi search (if available)
- `GET /restart/<service_name>` - Restart a specific service

### Service Status

The portal continuously monitors the health of both services:

- ✅ **Online**: Service is running and responding
- ❌ **Offline**: Service is not responding or failed to start
- ⏳ **Starting**: Service is in the process of starting up

## Troubleshooting

### Services Not Starting

1. **Check Dependencies**: Ensure all requirements are installed
   ```bash
   cd English && pip install -r requirements.txt
   cd ../Hindi && pip install -r requirements.txt
   ```

2. **Port Conflicts**: Make sure ports 3000, 5000, and 5500 are available
   ```bash
   lsof -i :3000
   lsof -i :5000
   lsof -i :5500
   ```

3. **Check Logs**: Look at the terminal output for specific error messages

4. **Manual Service Start**: Try starting services individually
   ```bash
   # Test English service
   cd English && python3 semantic_search_app.py
   
   # Test Hindi service (in another terminal)
   cd Hindi && python3 hindi_search_webapp.py
   ```

### Common Issues

- **Long Startup Time**: First-time startup may take longer due to model loading
- **Memory Usage**: Each service loads its own models, requiring significant RAM
- **File Permissions**: Ensure all Python files are readable and executable

## Configuration

### Service Ports

You can modify the service ports in `main_app.py`:

```python
SERVICES = {
    'english': {
        'port': 5000,  # Change this port
        # ...
    },
    'hindi': {
        'port': 5500,  # Change this port
        # ...
    }
}
```

### Service Scripts

The paths to the service scripts are configurable:

```python
SERVICES = {
    'english': {
        'script_path': 'English/semantic_search_app.py',
        # ...
    },
    'hindi': {
        'script_path': 'Hindi/hindi_search_webapp.py',
        # ...
    }
}
```

## Development

### Adding New Language Services

1. Create a new service directory
2. Implement the Flask application
3. Add the service configuration to `SERVICES` in `main_app.py`
4. Update the main template to include the new language option

### Customizing the UI

- Main template: `templates/main_index.html`
- Error template: `templates/service_error.html`
- Static assets: `static/` directory

## Dependencies

### Main Application
- Flask: Web framework
- requests: HTTP client for health checks

### Service Dependencies
Each service has its own requirements:
- **English Service**: See `English/requirements.txt`
- **Hindi Service**: See `Hindi/requirements.txt` (or uses English requirements)

## Performance

- **Startup Time**: 30-60 seconds (model loading)
- **Memory Usage**: 2-4 GB RAM (both services combined)
- **Concurrent Users**: Depends on individual service implementations

## License

This project is part of the Semantic Search NIC system. See individual service directories for specific licensing information.

## Support

For issues:
1. Check the troubleshooting section above
2. Look at terminal output for error messages
3. Verify all dependencies are correctly installed
4. Test individual services separately

## Version History

- **v1.0.0**: Initial release with English and Hindi service management
  - Automatic service startup and health monitoring
  - Modern responsive UI
  - Real-time status updates
  - Error handling and recovery
