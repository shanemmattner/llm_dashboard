import os
import logging
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from typing import List, Dict, Optional
from datetime import datetime

from document_manager import DocumentManager
from rag_utils import RAGManager

logger = logging.getLogger(__name__)

class RAGDashboard:
    """Dashboard for RAG document management and querying"""
    
    def __init__(self):
        """Initialize the dashboard with improved configuration"""
        # Initialize Dash app with enhanced settings
        self.app = Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                dbc.icons.FONT_AWESOME
            ],
            suppress_callback_exceptions=True,
            title="RAG Document Dashboard"
        )
        
        # Initialize managers
        try:
            self.doc_manager = DocumentManager()
            self.rag_manager = RAGManager("vectorstore")
            
            # Set up layout and callbacks
            self.app.layout = self._create_layout()
            self._register_callbacks()
            
            logger.info("Dashboard initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing dashboard: {e}")
            raise

    def _create_layout(self):
        """Create the dashboard layout with improved UI"""
        return dbc.Container([
            # Navigation Bar
            dbc.Navbar(
                dbc.Container([
                    dbc.NavbarBrand("RAG Document Dashboard", className="ms-2"),
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Documents", href="#documents")),
                        dbc.NavItem(dbc.NavLink("Query", href="#query")),
                        dbc.NavItem(dbc.NavLink("Stats", href="#stats"))
                    ])
                ]),
                color="primary",
                dark=True,
                className="mb-4"
            ),
            
            # System Status Alert
            html.Div(id='system-status', className="mb-4"),
            
            # Document Management Section
            dbc.Row([
                # RAG Documents Panel
                dbc.Col([
                    self._create_rag_panel()
                ], width=6),
                
                # Direct Files Panel
                dbc.Col([
                    self._create_direct_panel()
                ], width=6)
            ], className="mb-4"),
            
            # Query Section
            dbc.Row([
                dbc.Col([
                    self._create_query_section()
                ])
            ], className="mb-4"),
            
            # Stats Section
            dbc.Row([
                dbc.Col([
                    self._create_stats_section()
                ])
            ])
            
        ], fluid=True)

    def _create_rag_panel(self):
        """Create RAG documents panel"""
        return dbc.Card([
            dbc.CardHeader([
                html.H4("RAG Documents", className="d-inline"),
                dbc.Badge(
                    "RAG", 
                    color="primary",
                    className="ms-2"
                )
            ]),
            dbc.CardBody([
                # Upload component
                dcc.Upload(
                    id='rag-upload',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt me-2"),
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center'
                    },
                    multiple=True,
                    className="mb-3"
                ),
                
                # File list
                html.Div(id='rag-file-list', className="mb-3"),
                
                # Upload output
                html.Div(id='rag-upload-output', className="mb-3"),
                
                # Control buttons
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-trash me-2"), "Clear Store"],
                        id="clear-rag",
                        color="danger",
                        className="me-2"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-sync me-2"), "Refresh"],
                        id="refresh-rag",
                        color="secondary"
                    )
                ])
            ])
        ])

    def _create_direct_panel(self):
        """Create direct files panel"""
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Direct Context Files", className="d-inline"),
                dbc.Badge(
                    "Context", 
                    color="info",
                    className="ms-2"
                )
            ]),
            dbc.CardBody([
                # Upload component
                dcc.Upload(
                    id='direct-upload',
                    children=html.Div([
                        html.I(className="fas fa-file-upload me-2"),
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center'
                    },
                    multiple=True,
                    className="mb-3"
                ),
                
                # File list
                html.Div(id='direct-file-list', className="mb-3"),
                
                # Upload output
                html.Div(id='direct-upload-output', className="mb-3"),
                
                # Control buttons
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-trash me-2"), "Clear Files"],
                        id="clear-direct",
                        color="warning",
                        className="me-2"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-sync me-2"), "Refresh"],
                        id="refresh-direct",
                        color="secondary"
                    )
                ])
            ])
        ])

    def _create_query_section(self):
        """Create query section with improved UI"""
        return dbc.Card([
            dbc.CardHeader([
                html.H4("Query Documents", className="d-inline"),
                dbc.Badge(
                    "Query", 
                    color="success",
                    className="ms-2"
                )
            ]),
            dbc.CardBody([
                # Document Selection
                html.Div([
                    html.Label("Select Documents to Query:"),
                    dcc.Dropdown(
                        id='file-selector',
                        multi=True,
                        placeholder='Select specific files or leave empty to search all',
                        className="mb-2"
                    ),
                    dbc.FormText(
                        "Leave empty to search across all available documents"
                    ),
                    dcc.Checklist(
                        id='include-direct',
                        options=[{
                            'label': ' Include direct context files',
                            'value': 'yes'
                        }],
                        value=[],
                        className="mt-2"
                    )
                ], className="mb-3"),
                
                # Query Input
                html.Div([
                    html.Label("Enter Your Question:"),
                    dbc.InputGroup([
                        dbc.Input(
                            id='query-input',
                            type='text',
                            placeholder='What would you like to know?',
                            className="mb-2"
                        ),
                        dbc.InputGroupText(
                            html.I(className="fas fa-search")
                        )
                    ]),
                    dbc.FormText(
                        "Be specific in your question to get better results"
                    )
                ], className="mb-3"),
                
                # Query Controls
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-paper-plane me-2"), "Submit Query"],
                        id="submit-query",
                        color="primary",
                        className="me-2"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-eraser me-2"), "Clear"],
                        id="clear-query",
                        color="secondary"
                    )
                ], className="mb-3"),
                
                # Results Section
                dcc.Loading(
                    id="query-loading",
                    type="circle",
                    children=[
                        html.Div(id='query-output', className="mt-3")
                    ]
                )
            ])
        ])

    def _create_stats_section(self):
            """Create statistics section with detailed metrics"""
            return dbc.Card([
                dbc.CardHeader([
                    html.H4("System Statistics", className="d-inline"),
                    dbc.Badge(
                        "Stats", 
                        color="info",
                        className="ms-2"
                    )
                ]),
                dbc.CardBody([
                    dbc.Row([
                        # Document Stats
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Document Statistics"),
                                dbc.CardBody(id='document-stats')
                            ])
                        ], width=6),
                        
                        # Processing Stats
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Processing Statistics"),
                                dbc.CardBody(id='processing-stats')
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    
                    # Refresh Button
                    dbc.Button(
                        [html.I(className="fas fa-sync me-2"), "Refresh Stats"],
                        id="refresh-stats",
                        color="secondary",
                        className="mt-3"
                    )
                ])
            ])

    def _register_callbacks(self):
        """Register all dashboard callbacks with improved error handling"""
        
        @self.app.callback(
            Output('system-status', 'children'),
            Input('refresh-stats', 'n_clicks')
        )
        def update_system_status(n_clicks):
            """Update system status display"""
            try:
                rag_stats = self.rag_manager.get_document_stats()
                doc_stats = self.doc_manager.get_file_stats()
                
                return dbc.Alert([
                    html.H5("System Status", className="alert-heading"),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            html.P([
                                html.Strong("RAG Store: "),
                                f"{rag_stats['total_documents']} documents, ",
                                f"{rag_stats['total_chunks']} chunks"
                            ])
                        ]),
                        dbc.Col([
                            html.P([
                                html.Strong("Direct Files: "),
                                f"{doc_stats['prompt_files']} files"
                            ])
                        ])
                    ])
                ], color="info")
                
            except Exception as e:
                logger.error(f"Error updating system status: {e}")
                return dbc.Alert(
                    "Error retrieving system status",
                    color="danger"
                )

        @self.app.callback(
            Output('rag-upload-output', 'children'),
            Input('rag-upload', 'contents'),
            State('rag-upload', 'filename')
        )
        def handle_rag_upload(contents, filenames):
            """Handle RAG document uploads"""
            if not contents:
                return html.Div()
            
            outputs = []
            for content, filename in zip(contents, filenames):
                try:
                    success, filepath = self.doc_manager.save_file(
                        content, filename, 'rag'
                    )
                    
                    if success and self.rag_manager.process_file(filepath, filename):
                        outputs.append(
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-check-circle me-2"),
                                    f"Successfully processed {filename}"
                                ],
                                color="success",
                                className="mb-2"
                            )
                        )
                    else:
                        outputs.append(
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-exclamation-circle me-2"),
                                    f"Error processing {filename}"
                                ],
                                color="danger",
                                className="mb-2"
                            )
                        )
                except Exception as e:
                    outputs.append(
                        dbc.Alert(
                            [
                                html.I(className="fas fa-times-circle me-2"),
                                f"Error: {str(e)}"
                            ],
                            color="danger",
                            className="mb-2"
                        )
                    )
            
            return html.Div(outputs)

        @self.app.callback(
            Output('direct-upload-output', 'children'),
            Input('direct-upload', 'contents'),
            State('direct-upload', 'filename')
        )
        def handle_direct_upload(contents, filenames):
            """Handle direct file uploads"""
            if not contents:
                return html.Div()
            
            outputs = []
            for content, filename in zip(contents, filenames):
                try:
                    if self.doc_manager.load_prompt_file(content, filename):
                        outputs.append(
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-check-circle me-2"),
                                    f"Successfully loaded {filename}"
                                ],
                                color="success",
                                className="mb-2"
                            )
                        )
                    else:
                        outputs.append(
                            dbc.Alert(
                                [
                                    html.I(className="fas fa-exclamation-circle me-2"),
                                    f"Error loading {filename}"
                                ],
                                color="danger",
                                className="mb-2"
                            )
                        )
                except Exception as e:
                    outputs.append(
                        dbc.Alert(
                            [
                                html.I(className="fas fa-times-circle me-2"),
                                f"Error: {str(e)}"
                            ],
                            color="danger",
                            className="mb-2"
                        )
                    )
            
            return html.Div(outputs)

        @self.app.callback(
            [Output('file-selector', 'options'),
             Output('file-selector', 'value')],
            [Input('rag-upload-output', 'children'),
             Input('clear-rag', 'n_clicks')]
        )
        def update_file_selector(upload_output, clear_clicks):
            """Update file selector with improved metadata"""
            try:
                files = self.rag_manager.get_document_list()
                options = []
                
                for doc in files:
                    label = f"{doc['filename']}"
                    if doc.get('total_pages'):
                        label += f" ({doc['total_pages']} pages)"
                    if doc.get('chunk_count'):
                        label += f" - {doc['chunk_count']} chunks"
                        
                    options.append({
                        'label': label,
                        'value': doc['filepath']
                    })
                
                return options, []
                
            except Exception as e:
                logger.error(f"Error updating file selector: {e}")
                return [], []

        @self.app.callback(
            Output('query-output', 'children'),
            [Input('submit-query', 'n_clicks'),
             Input('clear-query', 'n_clicks')],
            [State('query-input', 'value'),
             State('file-selector', 'value'),
             State('include-direct', 'value')]
        )
        def handle_query(submit_clicks, clear_clicks, query, selected_files, include_direct):
            """Handle document querying with improved results display"""
            # Handle clear button
            if clear_clicks and clear_clicks > 0:
                return html.Div()
            
            # Handle query submission
            if not submit_clicks or not query:
                return html.Div()
            
            try:
                # Get direct context if requested
                context = ""
                if include_direct and 'yes' in include_direct:
                    context = self.doc_manager.get_prompt_context()
                
                # Query documents
                result = self.rag_manager.query_documents(
                    query=query,
                    rag_files=selected_files,
                    context_text=context
                )
                
                # Display results
                return html.Div([
                    # Answer section
                    dbc.Card([
                        dbc.CardHeader("Answer"),
                        dbc.CardBody(
                            dbc.Alert(result['result'], color="success")
                        )
                    ], className="mb-3"),
                    
                    # Sources section
                    dbc.Card([
                        dbc.CardHeader("Sources"),
                        dbc.CardBody([
                            html.Ul([
                                html.Li([
                                    html.Strong(os.path.basename(doc['source'])),
                                    f" (Page {doc['page']})" if 'page' in doc else "",
                                    html.Br(),
                                    html.Small(
                                        f"Relevance: {result['relevance_scores'][i]:.2%}",
                                        className="text-muted"
                                    )
                                ]) for i, doc in enumerate(result['sources'])
                            ])
                        ])
                    ], className="mb-3"),
                    
                    # Query stats
                    dbc.Card([
                        dbc.CardHeader("Query Statistics"),
                        dbc.CardBody([
                            html.P([
                                html.Strong("Context chunks used: "),
                                str(result['context_used'])
                            ]),
                            html.P([
                                html.Strong("Unique sources: "),
                                str(result.get('unique_sources', 'N/A'))
                            ])
                        ])
                    ])
                ])
                
            except Exception as e:
                logger.error(f"Query error: {e}")
                return dbc.Alert(
                    [
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        f"Error: {str(e)}"
                    ],
                    color="danger"
                )

        @self.app.callback(
            [Output('document-stats', 'children'),
             Output('processing-stats', 'children')],
            [Input('refresh-stats', 'n_clicks')]
        )
        def update_stats(n_clicks):
            """Update statistics displays"""
            try:
                rag_stats = self.rag_manager.get_document_stats()
                doc_stats = self.doc_manager.get_file_stats()
                
                # Document statistics
                doc_stats_content = html.Div([
                    html.P([
                        html.Strong("Total Documents: "),
                        str(rag_stats['total_documents'])
                    ]),
                    html.P([
                        html.Strong("Total Chunks: "),
                        str(rag_stats['total_chunks'])
                    ]),
                    html.P([
                        html.Strong("Document Types: "),
                        ", ".join(f"{k}: {v}" for k, v in rag_stats['document_types'].items())
                    ])
                ])
                
                # Processing statistics
                processing_stats_content = html.Div([
                    html.P([
                        html.Strong("Average Chunks per Document: "),
                        f"{rag_stats['avg_chunks_per_doc']:.1f}"
                    ]),
                    html.P([
                        html.Strong("Processing Dates: "),
                        ", ".join(rag_stats['processing_dates'][-5:])  # Show last 5 dates
                    ]),
                    html.P([
                        html.Strong("Storage Usage: "),
                        f"{doc_stats['summary']['total_storage_mb']} MB"
                    ])
                ])
                
                return doc_stats_content, processing_stats_content
                
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
                error_content = dbc.Alert(
                    "Error loading statistics",
                    color="danger"
                )
                return error_content, error_content

    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard application"""
        logger.info(f"Starting dashboard on port {port}")
        self.app.run_server(debug=debug, port=port)