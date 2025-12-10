from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from datetime import datetime
from data_quality import DataQualityInspector
from inference import initialize_predictor, get_fire_risk, get_predictor
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Basic configuration
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False

# Global predictor instance (initialized lazily)
_predictor = None

def get_or_init_predictor():
    """Get or initialize the global predictor instance."""
    global _predictor
    if _predictor is None:
        logger.info("Initializing wildfire predictor...")
        _predictor = initialize_predictor('model')
        logger.info("Predictor initialized successfully")
    return _predictor

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Wildfire Prediction API',
        'version': '1.0.0',
        'status': 'running'
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'wildfire-prediction-backend'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/validate_weather', methods=['POST'])
def validate_weather():
    """
    Validates incoming live weather data before inference.
    Returns quality status: 'safe', 'caution', or 'critical'.
    """
    try:
        payload = request.json
        if not payload:
            return jsonify({'error': 'Empty payload'}), 400

        # Convert dict/list to DataFrame for vectorised checks
        df = pd.DataFrame(payload) if isinstance(payload, list) else pd.DataFrame([payload])
        
        inspector = DataQualityInspector(df)
        inspector.run()
        
        # Determine status based on warnings
        has_warnings = inspector._issues_found 
        status = "caution" if has_warnings else "safe"
        
        return jsonify({
            'status': status,
            'report': inspector.report_lines,
            'meta': {
                'rows_processed': len(df),
                'columns_checked': list(df.columns)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'critical'}), 500


@app.route('/api/fire_risk', methods=['GET', 'POST'])
def fire_risk():
    """
    Get fire risk assessment for a given location and date.
    
    Accepts both GET and POST requests:
    - GET: /api/fire_risk?lat=25.29&lon=-80.65&date=2019-09-11
    - POST: {"latitude": 25.29, "longitude": -80.65, "date": "2019-09-11"}
    
    Returns:
    {
        "risk_level": "Low" | "Moderate" | "High" | "Extreme",
        "ml_prediction": true | false | null,
        "ml_probability": 0.0-1.0 | null,
        "fwi_level": "Low" | "Moderate" | "High" | "Very High" | "Extreme" | null,
        "fwi_value": float | null,
        "ground_truth": 0 | 1 | null,
        "elapsed_time_ms": float,
        "location": {"latitude": float, "longitude": float},
        "date": "YYYY-MM-DD"
    }
    """
    try:
        # Parse parameters from GET or POST
        if request.method == 'POST':
            data = request.json or {}
            lat = data.get('latitude') or data.get('lat')
            lon = data.get('longitude') or data.get('lon')
            date_str = data.get('date')
        else:
            lat = request.args.get('lat') or request.args.get('latitude')
            lon = request.args.get('lon') or request.args.get('longitude')
            date_str = request.args.get('date')
        
        # Validate parameters
        if lat is None or lon is None:
            return jsonify({
                'error': 'Missing required parameters: latitude and longitude',
                'usage': 'GET /api/fire_risk?lat=25.29&lon=-80.65&date=2019-09-11'
            }), 400
        
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Invalid latitude or longitude values. Must be numeric.'
            }), 400
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90):
            return jsonify({'error': 'Latitude must be between -90 and 90'}), 400
        if not (-180 <= lon <= 180):
            return jsonify({'error': 'Longitude must be between -180 and 180'}), 400
        
        # Parse date (default to today if not provided)
        if date_str:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({
                    'error': 'Invalid date format. Use YYYY-MM-DD.'
                }), 400
        else:
            date = datetime.now()
            date_str = date.strftime("%Y-%m-%d")
        
        # Get predictor
        predictor = get_or_init_predictor()
        
        # Get fire risk assessment
        result = get_fire_risk(lat, lon, date, predictor)
        
        # Add location and date to response
        result['location'] = {'latitude': lat, 'longitude': lon}
        result['date'] = date_str
        
        return jsonify(result)
    
    except Exception as e:
        logger.exception(f"Error in fire_risk endpoint: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/fire_risk/batch', methods=['POST'])
def fire_risk_batch():
    """
    Get fire risk assessment for multiple locations/dates.
    
    POST body:
    {
        "queries": [
            {"latitude": 25.29, "longitude": -80.65, "date": "2019-09-11"},
            {"latitude": 26.0, "longitude": -81.5, "date": "2019-09-12"}
        ]
    }
    
    Returns:
    {
        "results": [...],
        "total_elapsed_time_ms": float,
        "count": int
    }
    """
    try:
        data = request.json or {}
        queries = data.get('queries', [])
        
        if not queries:
            return jsonify({'error': 'No queries provided'}), 400
        
        if len(queries) > 100:
            return jsonify({'error': 'Maximum 100 queries per batch'}), 400
        
        # Get predictor
        predictor = get_or_init_predictor()
        
        results = []
        total_start = datetime.now()
        
        for query in queries:
            lat = query.get('latitude') or query.get('lat')
            lon = query.get('longitude') or query.get('lon')
            date_str = query.get('date')
            
            if lat is None or lon is None:
                results.append({'error': 'Missing latitude or longitude'})
                continue
            
            try:
                lat = float(lat)
                lon = float(lon)
                date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
                
                result = get_fire_risk(lat, lon, date, predictor)
                result['location'] = {'latitude': lat, 'longitude': lon}
                result['date'] = date_str or date.strftime("%Y-%m-%d")
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'location': {'latitude': lat, 'longitude': lon},
                    'date': date_str
                })
        
        total_elapsed = (datetime.now() - total_start).total_seconds() * 1000
        
        return jsonify({
            'results': results,
            'total_elapsed_time_ms': round(total_elapsed, 2),
            'count': len(results)
        })
    
    except Exception as e:
        logger.exception(f"Error in fire_risk_batch endpoint: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
