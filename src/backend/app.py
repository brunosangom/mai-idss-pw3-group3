from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from data_quality import DataQualityInspector
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Basic configuration
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
