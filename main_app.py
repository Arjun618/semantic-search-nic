#!/usr/bin/env python3
"""
Main application that provides language selection and manages both Hindi and English semantic search services
"""

import os
import sys
import time
import threading
import subprocess
from flask import Flask, render_template, redirect, url_for, jsonify
import requests
from pathlib import Path

app = Flask(__name__, template_folder='templates', static_folder='static')

# Service configuration
SERVICES = {
    'english': {
        'name': 'English Semantic Search',
        'port': 5000,
        'script_path': 'English/semantic_search_app.py',
        'url': 'http://localhost:5000',
        'description': 'Search NIC codes using natural language in English'
    },
    'hindi': {
        'name': 'Hindi Semantic Search', 
        'port': 5500,
        'script_path': 'Hindi/hindi_search_webapp.py',
        'url': 'http://localhost:5500',
        'description': 'हिंदी में प्राकृतिक भाषा का उपयोग करके NIC कोड खोजें'
    }
}

# Global variables to track service processes
service_processes = {}
service_status = {'english': False, 'hindi': False}

def check_service_health(service_name, max_retries=30, delay=2):
    """Check if a service is running and healthy"""
    service = SERVICES[service_name]
    url = service['url']
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {service['name']} is healthy on port {service['port']}")
                return True
        except requests.exceptions.RequestException:
            if attempt == 0:
                print(f"⏳ Waiting for {service['name']} to start...")
            time.sleep(delay)
    
    print(f"✗ {service['name']} failed to start after {max_retries * delay} seconds")
    return False

def start_service(service_name):
    """Start a service in a separate process"""
    service = SERVICES[service_name]
    script_path = os.path.join(os.path.dirname(__file__), service['script_path'])
    
    if not os.path.exists(script_path):
        print(f"✗ Script not found: {script_path}")
        return False
    
    try:
        print(f"🚀 Starting {service['name']} on port {service['port']}...")
        
        # Start the service process
        process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        service_processes[service_name] = process
        
        # Check if service started successfully
        if check_service_health(service_name):
            service_status[service_name] = True
            return True
        else:
            # If health check failed, terminate the process
            process.terminate()
            service_status[service_name] = False
            return False
            
    except Exception as e:
        print(f"✗ Failed to start {service['name']}: {str(e)}")
        service_status[service_name] = False
        return False

def start_all_services():
    """Start all services in parallel"""
    print("🌟 Starting Semantic Search Services...")
    
    # Start services in separate threads
    threads = []
    for service_name in SERVICES.keys():
        thread = threading.Thread(target=start_service, args=(service_name,))
        thread.start()
        threads.append(thread)
    
    # Wait for all services to start
    for thread in threads:
        thread.join()
    
    # Report status
    started_services = [name for name, status in service_status.items() if status]
    failed_services = [name for name, status in service_status.items() if not status]
    
    print(f"\n📊 Service Status:")
    for service_name, status in service_status.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {SERVICES[service_name]['name']}: {'Running' if status else 'Failed'}")
    
    if started_services:
        print(f"\n🎉 Successfully started {len(started_services)} service(s)")
    if failed_services:
        print(f"⚠️  Failed to start {len(failed_services)} service(s)")
    
    return len(started_services) > 0

@app.route('/')
def index():
    """Main landing page with language selection"""
    return render_template('main_index.html', services=SERVICES, service_status=service_status)

@app.route('/status')
def status():
    """API endpoint to check service status"""
    return jsonify({
        'services': service_status,
        'details': {
            name: {
                'running': status,
                'url': SERVICES[name]['url'],
                'port': SERVICES[name]['port']
            }
            for name, status in service_status.items()
        }
    })

@app.route('/english')
def redirect_to_english():
    """Redirect to English semantic search"""
    if service_status['english']:
        return redirect(SERVICES['english']['url'])
    else:
        return render_template('service_error.html', 
                             service_name='English Semantic Search',
                             error_message='Service is not running')

@app.route('/hindi')
def redirect_to_hindi():
    """Redirect to Hindi semantic search"""
    if service_status['hindi']:
        return redirect(SERVICES['hindi']['url'])
    else:
        return render_template('service_error.html', 
                             service_name='Hindi Semantic Search',
                             error_message='Service is not running')

@app.route('/restart/<service_name>')
def restart_service(service_name):
    """Restart a specific service"""
    if service_name not in SERVICES:
        return jsonify({'error': 'Invalid service name'}), 400
    
    # Stop existing process if running
    if service_name in service_processes:
        try:
            service_processes[service_name].terminate()
            service_processes[service_name].wait(timeout=10)
        except:
            pass
        del service_processes[service_name]
    
    # Start the service
    success = start_service(service_name)
    
    return jsonify({
        'service': service_name,
        'status': 'started' if success else 'failed',
        'running': service_status[service_name]
    })

def cleanup_services():
    """Clean up all running services"""
    print("\n🧹 Cleaning up services...")
    for service_name, process in service_processes.items():
        try:
            print(f"  Stopping {SERVICES[service_name]['name']}...")
            process.terminate()
            process.wait(timeout=10)
        except:
            try:
                process.kill()
            except:
                pass
    service_processes.clear()

if __name__ == '__main__':
    try:
        # Start all services
        if start_all_services():
            print(f"\n🌐 Main application starting on http://localhost:3000")
            print("🔗 Available services:")
            for name, service in SERVICES.items():
                if service_status[name]:
                    print(f"  • {service['name']}: {service['url']}")
            
            # Start the main Flask app
            app.run(debug=True, host='0.0.0.0', port=3000, use_reloader=False)
        else:
            print("❌ No services started successfully. Exiting.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Shutting down...")
    finally:
        cleanup_services()
        print("👋 Goodbye!")
