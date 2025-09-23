# setup_models.py
# Online-only version - no model downloads needed
import os
from pathlib import Path
print('Online-only mode enabled. No model downloads required.')
print('All AI processing will be done through online APIs.')
PROJECT_DIR = Path(__file__).parent
MODELS_DIR = Path(os.getenv('MODELS_DIR', PROJECT_DIR / 'models'))
print(f'Models directory: {MODELS_DIR}')
print('NOTE: Online API keys should be configured in environment variables.') 
