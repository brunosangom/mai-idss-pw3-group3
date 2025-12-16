from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime
from data_quality import DataQualityInspector
from inference import initialize_predictor, get_fire_risk, get_predictor, get_fire_risk_forecast
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

def log_batch_statistics(batch_type, total_requests, ml_used_count, fwi_used_count):
    """Log statistics about batch requests."""
    try:
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'batch_statistics.log')
        
        timestamp = datetime.now().isoformat()
        log_entry = f"{timestamp} - Type: {batch_type}, Total: {total_requests}, ML Used: {ml_used_count}, FWI Used: {fwi_used_count}\n"
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        logger.error(f"Failed to log batch statistics: {e}")

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
        
        # Generate request ID
        req_id = str(uuid.uuid4())
        
        # Get fire risk assessment
        result = get_fire_risk(lat, lon, date, predictor, source="fire_risk_endpoint", request_id=req_id)
        
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
        req_id = str(uuid.uuid4())
        
        ml_count = 0
        fwi_count = 0
        
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
                
                result = get_fire_risk(lat, lon, date, predictor, source="fire_risk_batch", request_id=req_id)
                result['location'] = {'latitude': lat, 'longitude': lon}
                result['date'] = date_str or date.strftime("%Y-%m-%d")
                results.append(result)
                
                if result.get('ml_prediction') is not None:
                    ml_count += 1
                if result.get('fwi_value') is not None:
                    fwi_count += 1
                    
            except Exception as e:
                results.append({
                    'error': str(e),
                    'location': {'latitude': lat, 'longitude': lon},
                    'date': date_str
                })
        
        log_batch_statistics('prediction_batch', len(queries), ml_count, fwi_count)
        
        total_elapsed = (datetime.now() - total_start).total_seconds() * 1000
        
        return jsonify({
            'results': results,
            'total_elapsed_time_ms': round(total_elapsed, 2),
            'count': len(results)
        })
    
    except Exception as e:
        logger.exception(f"Error in fire_risk_batch endpoint: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/fire_risk/forecast', methods=['GET', 'POST'])
def fire_risk_forecast():
    """
    Get fire risk forecast for a location for the next 8 days.
    
    GET: /api/fire_risk/forecast?lat=...&lon=...&date=...
    POST: {"latitude": ..., "longitude": ..., "date": ...}
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
            lat = data.get('latitude') or data.get('lat')
            lon = data.get('longitude') or data.get('lon')
            date_str = data.get('date')
        else:
            lat = request.args.get('lat') or request.args.get('latitude')
            lon = request.args.get('lon') or request.args.get('longitude')
            date_str = request.args.get('date')
            
        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
            
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid coordinates'}), 400
            
        start_date = None
        if date_str:
            try:
                start_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
            
        results = get_fire_risk_forecast(lat, lon, days=8, start_date=start_date)
        
        ml_count = sum(1 for r in results if r.get('ml_prediction') is not None)
        fwi_count = sum(1 for r in results if r.get('fwi_value') is not None)
        log_batch_statistics('forecast_single_loc', len(results), ml_count, fwi_count)
        
        return jsonify({
            'location': {'latitude': lat, 'longitude': lon},
            'forecast': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.exception(f"Error in fire_risk_forecast: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/fire_risk/history', methods=['GET', 'POST'])
def fire_risk_history():
    """
    Get fire risk history for a location for the past 7 days.
    
    GET: /api/fire_risk/history?lat=...&lon=...&date=...
    POST: {"latitude": ..., "longitude": ..., "date": ...}
    """
    try:
        if request.method == 'POST':
            data = request.json or {}
            lat = data.get('latitude') or data.get('lat')
            lon = data.get('longitude') or data.get('lon')
            date_str = data.get('date')
        else:
            lat = request.args.get('lat') or request.args.get('latitude')
            lon = request.args.get('lon') or request.args.get('longitude')
            date_str = request.args.get('date')
            
        if lat is None or lon is None:
            return jsonify({'error': 'Missing latitude or longitude'}), 400
            
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid coordinates'}), 400
            
        # Get predictor
        predictor = get_or_init_predictor()
        
        results = []
        req_id = str(uuid.uuid4())
        if date_str:
            try:
                today = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
        
        # Get history for past 7 days
        for i in range(7, -1, -1):
            date = today - pd.Timedelta(days=i)
            try:
                risk = get_fire_risk(lat, lon, date, predictor, source="fire_risk_history", request_id=req_id)
                risk['date'] = date.strftime("%Y-%m-%d")
                results.append(risk)
            except Exception as e:
                logger.warning(f"Failed to get history for {lat},{lon} on {date}: {e}")
                continue
                
        return jsonify({
            'location': {'latitude': lat, 'longitude': lon},
            'history': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.exception(f"Error in fire_risk_history: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/stations', methods=['GET'])
def get_stations():
    """
    Get list of weather stations from local CSV file - in production, this would use weather.gov API to fetch stations & automate updates.
    Returns a list of {name, lat, lon} objects.
    """
    try:
        # Load stations from local CSV file
        # Path relative to src/backend/app.py -> ../../data/selected_stations.csv
        csv_path = os.path.join(os.path.dirname(__file__), '../../data/selected_stations.csv')
        
        df = pd.read_csv(csv_path)
        
        all_stations = []
        for _, row in df.iterrows():
            try:
                # Parse geometry string
                geom = json.loads(row['geometry'])
                coords = geom['coordinates']
                lon, lat = coords[0], coords[1]
                
                all_stations.append({
                    'name': row['name'],
                    'lat': lat,
                    'lon': lon,
                    'id': row['stationIdentifier']
                })
            except Exception as e:
                logger.warning(f"Skipping station due to error: {e}")
                continue
            
        return jsonify({
            'count': len(all_stations),
            'stations': all_stations
        })
        
    except Exception as e:
        logger.exception(f"Error fetching stations: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/init', methods=['GET'])
def init_data():
    """
    Initialize application data: returns stations list from CSV.
    """
    return get_stations()


@app.route('/api/predict_all', methods=['GET'])
def predict_all():
    """
    Get predictions (forecast or history) for all stations.
    Query params:
    - mode: 'forecast' (default) or 'history'
    - date: YYYY-MM-DD (REQUIRED)
    """
    try:
        start_time = datetime.now()
        mode = request.args.get('mode', 'forecast')
        date_str = request.args.get('date')
        
        if not date_str:
            return jsonify({'error': 'Date parameter is required'}), 400
            
        try:
            base_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
        
        # Load stations
        csv_path = os.path.join(os.path.dirname(__file__), '../../data/selected_stations.csv')
        df = pd.read_csv(csv_path)
        
        stations = []
        for _, row in df.iterrows():
            try:
                geom = json.loads(row['geometry'])
                coords = geom['coordinates']
                stations.append({
                    'name': row['name'],
                    'lat': coords[1],
                    'lon': coords[0]
                })
            except:
                continue
        
        # Initialize predictor and ensure cache is loaded
        predictor = get_or_init_predictor()
        # Explicitly load data to satisfy "Wait until parquet cache is loaded"
        predictor.data_store._load_data()
        
        results = []
        req_id = str(uuid.uuid4())
        
        ml_count = 0
        fwi_count = 0
        total_predictions = 0
        
        # Process all stations
        for station in stations:
            lat, lon = station['lat'], station['lon']
            
            try:
                if mode == 'history':
                    # History logic (past 7 days relative to base_date)
                    station_history = []
                    for i in range(7, -1, -1):
                        date = base_date - pd.Timedelta(days=i)
                        try:
                            risk = get_fire_risk(lat, lon, date, predictor, source="predict_all_history", request_id=req_id)
                            risk['date'] = date.strftime("%Y-%m-%d")
                            station_history.append(risk)
                            
                            total_predictions += 1
                            if risk.get('ml_prediction') is not None:
                                ml_count += 1
                            if risk.get('fwi_value') is not None:
                                fwi_count += 1
                                
                        except Exception:
                            continue
                            
                    if station_history:
                        results.append({
                            'name': station['name'],
                            'location': {'latitude': lat, 'longitude': lon},
                            'history': station_history
                        })
                        
                else:
                    # Forecast logic (next 8 days relative to base_date)
                    station_forecast = []
                    for i in range(8):
                        target_date = base_date + pd.Timedelta(days=i)
                        try:
                            # Use get_fire_risk to leverage ML model + FWI
                            risk = get_fire_risk(lat, lon, target_date, predictor, source="predict_all_forecast", request_id=req_id)
                            risk['date'] = target_date.strftime("%Y-%m-%d")
                            station_forecast.append(risk)
                            
                            total_predictions += 1
                            if risk.get('ml_prediction') is not None:
                                ml_count += 1
                            if risk.get('fwi_value') is not None:
                                fwi_count += 1
                                
                        except Exception:
                            continue
                            
                    if station_forecast:
                        results.append({
                            'name': station['name'],
                            'location': {'latitude': lat, 'longitude': lon},
                            'forecast': station_forecast
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing station {station['name']}: {e}")
                continue
        
        log_batch_statistics(f'predict_all_{mode}', total_predictions, ml_count, fwi_count)
        
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Processed {len(results)} stations in {elapsed:.2f}ms")
        
        return jsonify({
            'results': results,
            'count': len(results),
            'elapsed_ms': elapsed
        })

    except Exception as e:
        logger.exception(f"Error in predict_all: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/historical-analysis')
def historical_analysis():
    try:
        # 1. Weather Trends (Pre-calculated)
        trends_path = os.path.join(os.path.dirname(__file__), '../../data/weather_trends.json')
        if os.path.exists(trends_path):
            with open(trends_path, 'r') as f:
                trends_data = json.load(f)
        else:
            trends_data = []

        # 2. Prediction Analysis (from station_predictions.csv)
        pred_path = os.path.join(os.path.dirname(__file__), '../../data/station_predictions.csv')
        
        prediction_accuracy = []
        error_trends = []
        hardest_stations = []
        
        if os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            
            # Accuracy per station over time
            pred_df['correct'] = (pred_df['prediction'] == pred_df['ground_truth']).astype(int)
            station_yearly = pred_df.groupby(['station_name', 'year'])['correct'].mean().reset_index()
            
            stations = station_yearly['station_name'].unique()
            # Limit to top 5 for clarity in line chart if too many
            for station in stations[:5]: 
                station_data = station_yearly[station_yearly['station_name'] == station]
                prediction_accuracy.append({
                    "id": station,
                    "data": [{"x": int(row['year']), "y": round(row['correct'] * 100, 1)} for _, row in station_data.iterrows()]
                })
                
            # FP/FN Trends
            pred_df['fp'] = ((pred_df['prediction'] == 1) & (pred_df['ground_truth'] == 0)).astype(int)
            pred_df['fn'] = ((pred_df['prediction'] == 0) & (pred_df['ground_truth'] == 1)).astype(int)
            
            global_errors = pred_df.groupby('year').agg({
                'fp': 'sum',
                'fn': 'sum',
                'station_name': 'count'
            }).reset_index()
            
            for _, row in global_errors.iterrows():
                total = row['station_name']
                error_trends.append({
                    "year": int(row['year']),
                    "fp_rate": round(row['fp'] / total * 100, 2),
                    "fn_rate": round(row['fn'] / total * 100, 2)
                })
                
            # Hardest Stations
            station_overall = pred_df.groupby('station_name')['correct'].mean().sort_values().head(5).reset_index()
            hardest_stations = [
                {"name": row['station_name'], "accuracy": round(row['correct'] * 100, 1)}
                for _, row in station_overall.iterrows()
            ]

        return jsonify({
            "weather_trends": trends_data,
            "prediction_accuracy": prediction_accuracy,
            "error_trends": error_trends,
            "hardest_stations": hardest_stations
        })

    except Exception as e:
        logger.error(f"Error in historical analysis: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
