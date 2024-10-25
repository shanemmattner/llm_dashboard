import logging
import os
from dashboard_layout import RAGDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Verify required environment variables are set"""
    required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

def main():
    """Main application entry point"""
    try:
        # Verify environment setup
        check_environment()
        
        # Initialize and run dashboard
        dashboard = RAGDashboard()
        dashboard.run(debug=True, port=8050)
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()