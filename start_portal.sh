#!/bin/bash

# Semantic Search Portal Startup Script
# This script starts the main application which manages both Hindi and English semantic search services

echo "🌟 Starting Semantic Search Portal..."
echo "=================================="

# Check if we're in the correct directory
if [[ ! -f "main_app.py" ]]; then
    echo "❌ Error: main_app.py not found. Please run this script from the project root directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Install main app requirements if needed
if [[ -f "requirements.txt" ]]; then
    echo "📦 Installing main application dependencies..."
    python3 -m pip install -r requirements.txt --quiet
fi

# Check if service dependencies are installed
echo "🔍 Checking service dependencies..."

# Check English service dependencies
if [[ -f "English/requirements.txt" ]]; then
    echo "  📦 Installing English service dependencies..."
    cd English
    python3 -m pip install -r requirements.txt --quiet
    cd ..
else
    echo "  ⚠️  English service requirements.txt not found"
fi

# Check Hindi service dependencies  
if [[ -f "Hindi/requirements.txt" ]] || [[ -f "English/requirements.txt" ]]; then
    echo "  📦 Installing Hindi service dependencies (using English requirements)..."
    # Hindi service can use English requirements as fallback
    cd Hindi
    if [[ -f "requirements.txt" ]]; then
        python3 -m pip install -r requirements.txt --quiet
    else
        python3 -m pip install -r ../English/requirements.txt --quiet
    fi
    cd ..
else
    echo "  ⚠️  Hindi service requirements not found"
fi

echo ""
echo "🚀 Starting Semantic Search Portal..."
echo "📊 Services that will be started:"
echo "  • Main Portal: http://localhost:3000"
echo "  • English Search: http://localhost:5000"
echo "  • Hindi Search: http://localhost:5500"
echo ""
echo "⏱️  Services may take a few moments to start up..."
echo "🌐 Once ready, open http://localhost:3000 in your browser"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=================================="

# Start the main application
python3 main_app.py
