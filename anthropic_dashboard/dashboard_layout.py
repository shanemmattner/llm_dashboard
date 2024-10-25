import dash_bootstrap_components as dbc
from dash import html, dcc

def create_layout():
    """Create the dashboard layout"""
    return dbc.Container([
        html.H1("RAG Dashboard", className="text-center my-4"),
        
        # File Upload Section
        dbc.Row([
            dbc.Col([
                html.H4("Upload Document"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or Select a File']),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='upload-output')
            ])
        ]),
        
        # Vector Store Contents Section
        dbc.Row([
            dbc.Col([
                html.H4("Vector Store Contents"),
                dbc.Button("Refresh", id="refresh-button", color="secondary", className="mb-2"),
                html.Div(id='vectorstore-contents'),
                dbc.Button("Clear Vector Store", id="clear-store-button", 
                          color="danger", className="mt-2"),
                html.Div(id='clear-store-output')
            ])
        ]),
        
        # File Selection and Query Section
        dbc.Row([
            dbc.Col([
                html.H4("Ask a Question"),
                dcc.Dropdown(
                    id='file-selector',
                    multi=True,
                    placeholder='Select files to query (optional)',
                    className="mb-2"
                ),
                dbc.Input(
                    id='query-input',
                    type='text',
                    placeholder='Enter your question...',
                    className="mb-2"
                ),
                dbc.Button("Submit", id="submit-button", color="primary", className="mb-2"),
                html.Div(id='answer-output', className="mt-4")
            ])
        ])
    ])