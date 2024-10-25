import os
import base64
import shutil
from pathlib import Path
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import logging
from typing import List, Dict

from rag_utils import RAGManager
from dashboard_layout import create_layout

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashApp:
    def __init__(self):
        self.UPLOAD_FOLDER = "uploads"
        self.VECTOR_DB_PATH = "vectorstore"
        self.CACHE_DIR = "cache/embeddings"
        
        # Create necessary directories with proper permissions
        for directory in [self.UPLOAD_FOLDER, self.VECTOR_DB_PATH, self.CACHE_DIR]:
            try:
                os.makedirs(directory, exist_ok=True)
                # Ensure directory has write permissions
                os.chmod(directory, 0o755)
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
                raise
        
        # Initialize Dash app
        self.app = Dash(
            __name__, 
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        logger.info("Initializing RAG Manager...")
        self.rag_manager = None
        try:
            self.rag_manager = RAGManager(self.VECTOR_DB_PATH)
            logger.info("RAG Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Manager: {e}")
            raise
        
        # Set layout
        self.app.layout = create_layout()
        self._register_callbacks()
    
    def _safe_remove_directory(self, path: str) -> bool:
        """Safely remove a directory and its contents"""
        try:
            if os.path.exists(path):
                # Ensure write permissions recursively
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        dirpath = os.path.join(root, d)
                        os.chmod(dirpath, 0o755)
                    for f in files:
                        filepath = os.path.join(root, f)
                        os.chmod(filepath, 0o644)
                
                shutil.rmtree(path)
                os.makedirs(path, exist_ok=True)
                return True
        except Exception as e:
            logger.error(f"Error removing directory {path}: {e}")
            return False
        return True

    def handle_upload(self, content: str, filename: str) -> html.Div:
        """Handle file upload"""
        if not content:
            return html.Div()
        
        try:
            logger.info(f"Processing upload for {filename}")
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            filepath = os.path.join(self.UPLOAD_FOLDER, filename)
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(decoded)
            os.chmod(filepath, 0o644)  # Ensure file is readable
            
            # Process file
            if self.rag_manager.process_file(filepath, filename):
                logger.info(f"Successfully processed {filename}")
                return html.Div(f"Successfully processed {filename}")
            else:
                logger.error(f"Failed to process {filename}")
                return html.Div(f"Error processing {filename}", style={'color': 'red'})
                
        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            return html.Div(f"Error: {str(e)}", style={'color': 'red'})

    def update_vectorstore_contents(self, n_clicks: int, upload_output: str, clear_output: str) -> tuple:
        """Update vector store contents display"""
        try:
            logger.debug("Fetching vector store contents")
            contents = self.rag_manager.get_vectorstore_contents()
            if not contents:
                return html.Div("No documents in vector store"), []
            
            # Create file list display
            file_list = html.Div([
                html.H5(f"Total Documents: {len(contents)}"),
                html.Ul([
                    html.Li(f"{os.path.basename(source)}: {info['count']} chunks")
                    for source, info in contents.items()
                ])
            ])
            
            # Create dropdown options
            options = [
                {'label': os.path.basename(source), 'value': source}
                for source in contents.keys()
            ]
            
            logger.debug(f"Found {len(contents)} documents in vector store")
            return file_list, options
            
        except Exception as e:
            logger.error(f"Error updating contents: {e}", exc_info=True)
            return html.Div("Error fetching vector store contents", style={'color': 'red'}), []

    def clear_vectorstore(self, n_clicks: int) -> html.Div:
        """Clear vector store and related directories"""
        if not n_clicks:
            return html.Div()
        
        try:
            logger.info("Starting vector store clear")
            
            # Clear the vector store
            if self.rag_manager.clear_vectorstore():
                # Clean up directories
                directories_cleared = all([
                    self._safe_remove_directory(self.UPLOAD_FOLDER),
                    self._safe_remove_directory(self.VECTOR_DB_PATH),
                    self._safe_remove_directory(self.CACHE_DIR)
                ])
                
                if directories_cleared:
                    logger.info("Vector store and directories cleared successfully")
                    # Reinitialize RAG manager
                    self.rag_manager = RAGManager(self.VECTOR_DB_PATH)
                    return html.Div("Vector store cleared successfully", style={'color': 'green'})
                else:
                    raise Exception("Failed to clear all directories")
            else:
                logger.error("Failed to clear vector store")
                return html.Div("Error clearing vector store", style={'color': 'red'})
                
        except Exception as e:
            logger.error(f"Clear store error: {e}", exc_info=True)
            return html.Div(f"Error: {str(e)}", style={'color': 'red'})

    def answer_question(self, n_clicks: int, query: str, selected_files: List[str]) -> html.Div:
        """Process a query and return answer"""
        if not n_clicks or not query:
            return html.Div()
        
        try:
            logger.info(f"Processing query: {query}")
            logger.info(f"Selected files: {selected_files}")
            
            result = self.rag_manager.query_documents(query, selected_files)
            
            # Create response with timing information
            response = html.Div([
                html.H5("Answer:"),
                html.P(result['result']),
                html.H5("Sources:"),
                html.Ul([
                    html.Li([
                        f"Source: {os.path.basename(doc.metadata['source'])} ",
                        f"{'Page: ' + str(doc.metadata['page']) if 'page' in doc.metadata else ''}"
                    ])
                    for doc in result['source_documents']
                ]),
                html.Div([
                    f"Query time: {result.get('query_time', 0):.2f}s, ",
                    f"LLM time: {result.get('llm_time', 0):.2f}s"
                ], className="text-muted")
            ])
            
            return response
            
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return html.Div(f"Error: {str(e)}", style={'color': 'red'})

    def _register_callbacks(self):
        self.app.callback(
            Output('upload-output', 'children'),
            Input('upload-data', 'contents'),
            State('upload-data', 'filename')
        )(self.handle_upload)
        
        self.app.callback(
            [Output('vectorstore-contents', 'children'),
             Output('file-selector', 'options')],
            [Input('refresh-button', 'n_clicks'),
             Input('upload-output', 'children'),
             Input('clear-store-output', 'children')]
        )(self.update_vectorstore_contents)
        
        self.app.callback(
            Output('clear-store-output', 'children'),
            Input('clear-store-button', 'n_clicks')
        )(self.clear_vectorstore)
        
        self.app.callback(
            Output('answer-output', 'children'),
            Input('submit-button', 'n_clicks'),
            [State('query-input', 'value'),
             State('file-selector', 'value')]
        )(self.answer_question)

    def run(self, debug=True, port=8050):
        self.app.run_server(debug=debug, port=port)

if __name__ == '__main__':
    app = DashApp()
    app.run()