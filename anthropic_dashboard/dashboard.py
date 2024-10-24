# Standard library imports
import os
from pathlib import Path
import uuid
import sqlite3
import base64

# Dash related imports
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# LangChain related imports
from langchain_community.vectorstores import Chroma  # Updated import
from langchain_openai import OpenAIEmbeddings      # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader  # Updated import
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic

# Rest of the code remains the same...
import anthropic

# Initialize the Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Configure environment variables and paths
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_PATH = "files.db"
UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "vectorstore"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS files
                 (id TEXT PRIMARY KEY, filename TEXT, filepath TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Initialize LangChain components
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
llm = ChatAnthropic(model="claude-3-sonnet-20240229", anthropic_api_key=ANTHROPIC_API_KEY)

# Layout components
file_upload = dcc.Upload(
    id='upload-data',
    children=html.Div([
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
        'textAlign': 'center',
        'margin': '10px'
    },
    multiple=True
)

query_input = dbc.Input(
    id='query-input',
    type='text',
    placeholder='Enter your question...',
    style={'margin': '10px'}
)

# Layout
app.layout = dbc.Container([
    html.H1("RAG Dashboard with Claude", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            html.H4("Upload Documents"),
            file_upload,
            html.Div(id='file-list')
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("Ask Questions"),
            query_input,
            dbc.Button("Submit", id="submit-button", color="primary", className="mt-2"),
            html.Div(id='answer-output', className="mt-4")
        ], width=12)
    ])
])

# Callbacks
@app.callback(
    Output('file-list', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_files(contents, filenames):
    if contents is None:
        return []
    
    file_list = []
    for content, filename in zip(contents, filenames):
        # Save file
        file_id = str(uuid.uuid4())
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        
        filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        with open(filepath, 'wb') as f:
            f.write(decoded)
        
        # Store in SQLite
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO files (id, filename, filepath) VALUES (?, ?, ?)",
                 (file_id, filename, filepath))
        conn.commit()
        conn.close()
        
        # Process for RAG
        loader = TextLoader(filepath)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore.add_documents(splits)
        
        file_list.append(html.Div(f"Uploaded: {filename}"))
    
    return file_list

@app.callback(
    Output('answer-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('query-input', 'value')
)
def answer_question(n_clicks, query):
    if n_clicks is None or query is None:
        return ""
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Get answer
    result = qa_chain({"query": query})
    
    return html.Div([
        html.H5("Answer:"),
        html.P(result['result']),
        html.H5("Sources:"),
        html.Ul([html.Li(str(doc.metadata)) for doc in result['source_documents']])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)