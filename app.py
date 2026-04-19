"""
Flask Backend for Crime Analytics Dashboard
Serves the HTML frontend and provides API endpoints for model predictions
Integrates pre-trained models with analysis outputs and visualizations
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import warnings
import webbrowser
import threading
from pathlib import Path
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='Frontend', static_url_path='/static')
CORS(app)

# Directories
MODELS_DIR = 'models'
OUTPUT_DIR = 'Output'
FRONTEND_DIR = 'Frontend'

# Load pre-trained models
MODELS = {}
OUTPUT_DATA = {}

def load_models():
    """Load all trained models"""
    global MODELS
    try:
        MODELS['juvenile_rf'] = joblib.load(os.path.join(MODELS_DIR, 'juvenile_rf_model.pkl'))
        MODELS['juvenile_gb'] = joblib.load(os.path.join(MODELS_DIR, 'juvenile_gb_model.pkl'))
        MODELS['juvenile_features'] = joblib.load(os.path.join(MODELS_DIR, 'juvenile_features.pkl'))
        MODELS['ews_lr'] = joblib.load(os.path.join(MODELS_DIR, 'ews_lr_model.pkl'))
        MODELS['ews_features'] = joblib.load(os.path.join(MODELS_DIR, 'ews_features.pkl'))
        MODELS['victim_profiles'] = joblib.load(os.path.join(MODELS_DIR, 'victim_profiles.pkl'))
        MODELS['victim_state_vulnerability'] = joblib.load(os.path.join(MODELS_DIR, 'victim_state_vulnerability.pkl'))
        print("✓ All models loaded successfully")
        return True
    except Exception as e:
        print(f"⚠️  Error loading models: {e}")
        return False

def load_output_data():
    """Load analysis outputs and results"""
    global OUTPUT_DATA
    try:
        # Load CSVs
        csv_files = {
            'juvenile_risk': 'juvenile_recidivism_state_risk.csv',
            'ews_risk': 'institutional_stress_state_risk.csv',
            'victim_vulnerability': 'victim_state_vulnerability.csv',
            'victim_profiles': 'victim_vulnerability_profiles.csv',
            'victim_trends': 'victim_temporal_trends.csv',
            'ews_features': 'ews_feature_importance.csv',
            'intervention': 'intervention_recommendations.csv',
            'demographic_shifts': 'victim_demographic_shifts.csv'
        }
        
        for key, filename in csv_files.items():
            filepath = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(filepath):
                OUTPUT_DATA[key] = pd.read_csv(filepath)
                print(f"✓ Loaded {key}: {len(OUTPUT_DATA[key])} rows")
        
        # List available images
        image_dir = Path(OUTPUT_DIR)
        OUTPUT_DATA['images'] = {
            'png': [f.name for f in image_dir.glob('*.png')],
            'dir': OUTPUT_DIR
        }
        print(f"✓ Found {len(OUTPUT_DATA['images']['png'])} visualization images")
        
        return True
    except Exception as e:
        print(f"⚠️  Error loading output data: {e}")
        return False

def open_browser():
    """Open browser after a short delay to allow server to start"""
    import time
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

# Initialize on startup
print("\n" + "="*80)
print("🚀 LOADING MODELS AND DATA")
print("="*80)
load_models()
load_output_data()
print("="*80 + "\n")

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    """Serve the home overview page"""
    return send_from_directory(FRONTEND_DIR, 'india_crime_analytics_dashboard.html')

@app.route('/home')
def home():
    """Home Overview Dashboard"""
    return send_from_directory(FRONTEND_DIR, 'india_crime_analytics_dashboard.html')

@app.route('/institutional')
def institutional():
    """Institutional Stress & EWS Dashboard"""
    return send_from_directory(FRONTEND_DIR, 'india_crime_analytics_dashboard.html')

@app.route('/juvenile')
def juvenile():
    """Juvenile Recidivism Dashboard"""
    return send_from_directory(FRONTEND_DIR, 'india_crime_analytics_dashboard.html')

@app.route('/victim')
def victim():
    """Victim Vulnerability Dashboard"""
    return send_from_directory(FRONTEND_DIR, 'india_crime_analytics_dashboard.html')

@app.route('/page/<page_name>')
def serve_page(page_name):
    """Serve HTML pages by name"""
    page_map = {
        'institutional': '1.html',
        'juvenile': '2.html',
        'victim': '3.html',
        'analytics': 'code.html',
        '1': '1.html',
        '2': '2.html',
        '3': '3.html',
        'code': 'code.html'
    }
    
    filename = page_map.get(page_name.lower(), '1.html')
    return send_from_directory(FRONTEND_DIR, filename)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files from Frontend"""
    return send_from_directory(FRONTEND_DIR, path)

@app.route('/output/<path:path>')
def serve_output(path):
    """Serve images and data from Output folder"""
    return send_from_directory(OUTPUT_DIR, path)

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve visualization images"""
    return send_from_directory(OUTPUT_DIR, filename)

# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(MODELS) > 0,
        'data_loaded': len(OUTPUT_DATA) > 0,
        'models_count': len([m for m in MODELS.keys() if not m.endswith('features')]),
        'visualizations': len(OUTPUT_DATA.get('images', {}).get('png', []))
    })

@app.route('/api/dashboard/statistics', methods=['GET'])
def dashboard_statistics():
    """Get overall dashboard statistics"""
    try:
        stats = {
            'models_trained': len([m for m in MODELS.keys() if not m.endswith('features')]),
            'visualizations_generated': len(OUTPUT_DATA.get('images', {}).get('png', [])),
            'analysis_reports': len([k for k in OUTPUT_DATA.keys() if k != 'images']),
            'states_analyzed': 28,
            'data_years': '2001-2010',
            'last_updated': 'Training Complete'
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/juvenile', methods=['POST'])
def predict_juvenile():
    """
    Predict juvenile recidivism
    Expected input: state_data dict with feature values
    """
    try:
        data = request.json
        
        if not MODELS.get('juvenile_rf'):
            return jsonify({'error': 'Juvenile model not loaded'}), 500
        
        # Use Random Forest model
        model = MODELS['juvenile_rf']
        features = MODELS['juvenile_features']
        
        # Create feature vector from input
        X = np.zeros((1, len(features)))
        for i, feat in enumerate(features):
            X[0, i] = data.get(feat, 0)
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability_no_recidivism': float(probability[0]),
            'probability_recidivism': float(probability[1]),
            'risk_level': 'High' if prediction == 1 else 'Low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/ews', methods=['POST'])
def predict_ews():
    """
    Predict institutional stress (Early Warning System)
    Expected input: stress_data dict with complaint metrics
    """
    try:
        data = request.json
        
        if not MODELS.get('ews_lr'):
            return jsonify({'error': 'EWS model not loaded'}), 500
        
        model = MODELS['ews_lr']
        features = MODELS['ews_features']
        
        # Create feature vector
        X = np.zeros((1, len(features)))
        for i, feat in enumerate(features):
            X[0, i] = data.get(feat, 0)
        
        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability_low_risk': float(probability[0]),
            'probability_high_risk': float(probability[1]),
            'risk_level': 'High' if prediction == 1 else 'Low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualizations/list', methods=['GET'])
def list_visualizations():
    """Get list of available visualization images"""
    try:
        images = OUTPUT_DATA.get('images', {}).get('png', [])
        return jsonify({
            'total': len(images),
            'images': images,
            'base_url': '/output/'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualizations/juvenile', methods=['GET'])
def get_juvenile_visualizations():
    """Get juvenile recidivism analysis visualizations"""
    try:
        images = {
            'feature_importance': '/output/juvenile_recidivism_feature_importance.png',
            'confusion_matrix': '/output/juvenile_recidivism_confusion_matrix.png'
        }
        
        # Get risk data
        risk_data = OUTPUT_DATA.get('juvenile_risk', pd.DataFrame())
        
        return jsonify({
            'images': images,
            'data': {
                'columns': risk_data.columns.tolist(),
                'rows': risk_data.head(10).values.tolist() if not risk_data.empty else [],
                'total_records': len(risk_data)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualizations/ews', methods=['GET'])
def get_ews_visualizations():
    """Get Institutional Stress (EWS) visualizations"""
    try:
        images = {
            'feature_importance': '/output/institutional_stress_feature_importance.png',
            'roc_curve': '/output/institutional_stress_roc_curve.png',
            'risk_distribution': '/output/institutional_stress_risk_distribution.png'
        }
        
        # Get risk data and feature importance
        risk_data = OUTPUT_DATA.get('ews_risk', pd.DataFrame())
        feature_data = OUTPUT_DATA.get('ews_features', pd.DataFrame())
        
        return jsonify({
            'images': images,
            'risk_data': {
                'columns': risk_data.columns.tolist(),
                'rows': risk_data.head(10).values.tolist() if not risk_data.empty else [],
                'total_records': len(risk_data)
            },
            'feature_importance': {
                'columns': feature_data.columns.tolist(),
                'rows': feature_data.head(10).values.tolist() if not feature_data.empty else [],
                'total_features': len(feature_data)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/visualizations/victim', methods=['GET'])
def get_victim_visualizations():
    """Get victim vulnerability analysis visualizations"""
    try:
        images = {
            'heatmap': '/output/victim_vulnerability_heatmap.png',
            'temporal_trends': '/output/victim_temporal_trends.png',
            'comparative_risk': '/output/comparative_risk_analysis.png'
        }
        
        # Get victim data
        vuln_data = OUTPUT_DATA.get('victim_vulnerability', pd.DataFrame())
        profiles_data = OUTPUT_DATA.get('victim_profiles', pd.DataFrame())
        trends_data = OUTPUT_DATA.get('victim_trends', pd.DataFrame())
        
        return jsonify({
            'images': images,
            'vulnerability': {
                'columns': vuln_data.columns.tolist(),
                'rows': vuln_data.head(10).values.tolist() if not vuln_data.empty else [],
                'total_records': len(vuln_data)
            },
            'profiles': {
                'columns': profiles_data.columns.tolist(),
                'rows': profiles_data.head(10).values.tolist() if not profiles_data.empty else [],
                'total_records': len(profiles_data)
            },
            'trends': {
                'columns': trends_data.columns.tolist(),
                'rows': trends_data.values.tolist() if not trends_data.empty else []
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analysis/juvenile', methods=['GET'])
def get_juvenile_analysis():
    """Get full juvenile recidivism analysis"""
    try:
        df = OUTPUT_DATA.get('juvenile_risk', pd.DataFrame())
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        return jsonify({
            'data': df.to_dict('records'),
            'summary': {
                'total_records': len(df),
                'columns': df.columns.tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analysis/ews', methods=['GET'])
def get_ews_analysis():
    """Get full EWS analysis"""
    try:
        risk_df = OUTPUT_DATA.get('ews_risk', pd.DataFrame())
        feature_df = OUTPUT_DATA.get('ews_features', pd.DataFrame())
        
        if risk_df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        return jsonify({
            'risk_assessment': risk_df.to_dict('records'),
            'feature_importance': feature_df.to_dict('records') if not feature_df.empty else [],
            'summary': {
                'total_states': len(risk_df),
                'high_risk_states': len(risk_df[risk_df.get('Risk_Score', 0) > 0.5]) if 'Risk_Score' in risk_df.columns else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analysis/victim', methods=['GET'])
def get_victim_analysis():
    """Get full victim vulnerability analysis"""
    try:
        vuln_df = OUTPUT_DATA.get('victim_vulnerability', pd.DataFrame())
        profiles_df = OUTPUT_DATA.get('victim_profiles', pd.DataFrame())
        
        if vuln_df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        return jsonify({
            'state_vulnerability': vuln_df.to_dict('records'),
            'victim_profiles': profiles_df.to_dict('records') if not profiles_df.empty else [],
            'summary': {
                'total_states': len(vuln_df),
                'total_victims': int(vuln_df.get('Total_Vulnerability', pd.Series()).sum()) if 'Total_Vulnerability' in vuln_df.columns else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get intervention recommendations"""
    try:
        df = OUTPUT_DATA.get('intervention', pd.DataFrame())
        if df.empty:
            return jsonify({'error': 'No recommendations available'}), 404
        
        return jsonify(df.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/victim-vulnerability', methods=['GET'])
def get_victim_vulnerability():
    """
    Get victim vulnerability data for visualization
    """
    try:
        if 'victim_state_vulnerability' not in MODELS:
            return jsonify({'error': 'Victim data not loaded'}), 500
        
        data = MODELS['victim_state_vulnerability']
        
        # Return top 15 states
        top_states = data.nlargest(15, 'Total_Vulnerability')
        
        return jsonify({
            'states': top_states.index.tolist(),
            'murder': top_states.get('Murder', pd.Series()).tolist(),
            'culpable_homicide': top_states.get('Culpable Homicide', pd.Series()).tolist(),
            'total': top_states['Total_Vulnerability'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get dashboard statistics
    """
    try:
        stats = {
            'models_trained': len([m for m in MODELS.keys() if not m.endswith('features')]),
            'data_points_processed': 3125,  # Approximate based on training data
            'predictions_available': 3,
            'states_covered': 28
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data/all', methods=['GET'])
def get_all_data():
    """
    Get all analysis data in structured format for frontend
    Returns combined data from all three models
    """
    try:
        juvenile_df = OUTPUT_DATA.get('juvenile_risk', pd.DataFrame())
        ews_df = OUTPUT_DATA.get('ews_risk', pd.DataFrame())
        victim_df = OUTPUT_DATA.get('victim_state_vulnerability', pd.DataFrame())
        
        # Convert dataframes to dict records
        juvenile_data = juvenile_df.to_dict('records') if not juvenile_df.empty else []
        ews_data = ews_df.to_dict('records') if not ews_df.empty else []
        victim_data = victim_df.to_dict('records') if not victim_df.empty else []
        
        return jsonify({
            'juvenile': juvenile_data,
            'ews': ews_data,
            'victim': victim_data,
            'metadata': {
                'total_states_juvenile': len(juvenile_data),
                'total_states_ews': len(ews_data),
                'total_states_victim': len(victim_data),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        })
    except Exception as e:
        print(f"Error in /api/data/all: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/data/juvenile-top-states', methods=['GET'])
def get_juvenile_top_states():
    """Get top N states by juvenile recidivism risk"""
    try:
        df = OUTPUT_DATA.get('juvenile_risk', pd.DataFrame())
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        n = request.args.get('limit', default=15, type=int)
        
        # Get top states
        top_states = df.nlargest(n, 'Risk_Score') if 'Risk_Score' in df.columns else df.head(n)
        
        return jsonify({
            'states': top_states.to_dict('records'),
            'count': len(top_states)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data/ews-top-states', methods=['GET'])
def get_ews_top_states():
    """Get top N states by EWS risk"""
    try:
        df = OUTPUT_DATA.get('ews_risk', pd.DataFrame())
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        n = request.args.get('limit', default=15, type=int)
        
        # Get top states
        top_states = df.nlargest(n, 'Risk_Score') if 'Risk_Score' in df.columns else df.head(n)
        
        return jsonify({
            'states': top_states.to_dict('records'),
            'count': len(top_states)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/data/victim-top-states', methods=['GET'])
def get_victim_top_states():
    """Get top N states by victim vulnerability"""
    try:
        df = OUTPUT_DATA.get('victim_state_vulnerability', pd.DataFrame())
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        n = request.args.get('limit', default=15, type=int)
        
        # Get top states (assuming vulnerability score column exists)
        if 'Risk_Score' in df.columns:
            top_states = df.nlargest(n, 'Risk_Score')
        else:
            top_states = df.head(n)
        
        return jsonify({
            'states': top_states.to_dict('records'),
            'count': len(top_states)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors - serve home page for SPA routing"""
    return send_from_directory(FRONTEND_DIR, 'india_crime_analytics_dashboard.html')

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    print("\n" + "="*100)
    print("🚀 CRIME ANALYTICS DASHBOARD - FLASK BACKEND")
    print("="*100)
    print("\n✓ Frontend: http://localhost:5000")
    print("✓ API Base: http://localhost:5000/api")
    print("✓ Output Images: /output/<filename>")
    print("\n✓ Available Pages:")
    print("  - http://localhost:5000/ (Institutional Stress Dashboard)")
    print("  - http://localhost:5000/juvenile (Juvenile Recidivism Dashboard)")
    print("  - http://localhost:5000/victim (Victim Vulnerability Dashboard)")
    print("  - http://localhost:5000/analytics (Crime Data Analytics)")
    
    print("\n✓ Available API Endpoints:")
    print("  - /api/health - Health check")
    print("  - /api/dashboard/statistics - Dashboard stats")
    print("  - /api/visualizations/list - List all visualizations")
    print("  - /api/visualizations/juvenile - Juvenile analysis with images")
    print("  - /api/visualizations/ews - EWS analysis with images")
    print("  - /api/visualizations/victim - Victim analysis with images")
    print("  - /api/analysis/juvenile - Full juvenile data")
    print("  - /api/analysis/ews - Full EWS data")
    print("  - /api/analysis/victim - Full victim data")
    print("  - /api/recommendations - Intervention recommendations")
    print("  - /api/data/all - All model data combined")
    print("  - /api/data/juvenile-top-states - Top juvenile risk states")
    print("  - /api/data/ews-top-states - Top EWS risk states")
    print("  - /api/data/victim-top-states - Top victim vulnerability states")
    
    print("\n" + "="*100)
    print("Opening browser in 2 seconds...")
    print("="*100 + "\n")
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    app.run(debug=True, host='localhost', port=5000, use_reloader=False)
