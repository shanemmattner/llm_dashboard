# Standard library imports
import os
from pathlib import Path
import uuid
import base64
import shutil
import logging
from pymongo import MongoClient
import dash

# Dash related imports
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

# LangChain related imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic
import fitz  # PyMuPDF for PDF processing
from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document

# Background task imports
import threading

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Dash app with Bootstrap theme and suppress callback exceptions
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Configure environment variables and paths
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
UPLOAD_FOLDER = "uploads"
VECTOR_DB_PATH = "vectorstore"

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client['filedb']
file_collection = db['files']

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Initialize LangChain components
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", anthropic_api_key=ANTHROPIC_API_KEY)


logging.basicConfig(level=logging.INFO)  # Change from DEBUG to INFO or WARNING
logger = logging.getLogger(__name__)

# Helper function to process files in background
def process_file(filepath, file_id, filename):
    logger.info(f"Processing file {filename} with ID {file_id} at {filepath}")
    _, file_extension = os.path.splitext(filename)
    
    if file_extension.lower() == ".pdf":
        loader = PDFLoader(filepath)
    else:
        loader = TextLoader(filepath)

    try:
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {filename}")
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return

    document_texts = []
    for doc in documents:
        if isinstance(doc, str):
            document_texts.append(doc)
        elif isinstance(doc, Document):
            document_texts.append(doc.page_content)
        else:
            logger.warning(f"Unexpected document format: {type(doc)}")
            continue

    metadata = {
        'file_id': file_id,
        'filename': filename,
        'path': filepath,
        'chunk_count': len(document_texts)
    }
    
    file_collection.insert_one(metadata)
    logger.info(f"Inserted metadata for {filename} into MongoDB")

    document_objects = [Document(page_content=text, metadata={"source": filename}) for text in document_texts]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document_objects)
    vectorstore.add_documents(splits)
    logger.info(f"Added {len(splits)} chunks to vectorstore for {filename}")

# PDFLoader class to handle PDF extraction
class PDFLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        documents = []
        with fitz.open(self.file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text = page.get_text("text")
                documents.append(text)
        return documents


# Function to get the list of files already uploaded to MongoDB
def get_uploaded_files():
    files = list(file_collection.find())
    return files


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

delete_file_input = dbc.Input(
    id='file-to-delete',
    type='text',
    placeholder='Enter file ID to delete',
    style={'margin': '10px'}
)

delete_file_button = dbc.Button(
    "Delete File",
    id='delete-file-button',
    color="danger",
    className="mt-2"
)

# Layout
app.layout = dbc.Container([
    html.H1("RAG Dashboard with Claude", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.H4("Upload Documents"),
            file_upload,
            html.Div(id='file-list')  # This will display uploaded files with checkboxes
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.H4("Delete Document"),
            delete_file_input,
            delete_file_button,
            html.Div(id='delete-output')
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
# Callbacks
@app.callback(
    Output('file-list', 'children'),
    [Input('upload-data', 'contents'),
     Input('delete-file-button', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('file-to-delete', 'value')]
)
def manage_files(contents, delete_clicks, filenames, file_id_to_delete):
    ctx = dash.callback_context

    # Get the list of files from MongoDB
    uploaded_files = get_uploaded_files()

    # Handle initial layout load when nothing has triggered yet
    if not ctx.triggered:
        return [
            html.H5("Select Documents for Query:"),
            dcc.Checklist(
                id='file-checklist',
                options=[
                    {'label': f"{file['filename']} (ID: {file['file_id']})", 'value': file['file_id']}
                    for file in uploaded_files
                ],
                value=[]  # Initially, no files selected
            )
        ]

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    file_list = []

    # Handle file uploads
    if triggered_id == 'upload-data':
        if contents is None:
            return []

        for content, filename in zip(contents, filenames):
            try:
                file_id = str(uuid.uuid4())
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
                with open(filepath, 'wb') as f:
                    f.write(decoded)

                # Process file in a background thread
                thread = threading.Thread(target=process_file, args=(filepath, file_id, filename))
                thread.start()

                file_list.append(html.Div(f"Uploaded: {filename} (Processing in background)"))

            except Exception as e:
                file_list.append(html.Div(f"Failed to upload {filename}: {str(e)}"))

    # Handle file deletion
    elif triggered_id == 'delete-file-button':
        if delete_clicks is None or file_id_to_delete is None:
            return []

        file_data = file_collection.find_one({"file_id": file_id_to_delete})
        if file_data:
            try:
                os.remove(file_data['path'])
                file_collection.delete_one({"file_id": file_id_to_delete})
                # Add logic to remove the file from the vectorstore

                file_list.append(html.Div(f"File {file_data['filename']} deleted."))
            except Exception as e:
                file_list.append(html.Div(f"Failed to delete file: {str(e)}"))
        else:
            file_list.append(html.Div(f"File ID {file_id_to_delete} not found."))

    # Update the file list with checkboxes
    uploaded_files = get_uploaded_files()
    return [
        html.H5("Select Documents for Query:"),
        dcc.Checklist(
            id='file-checklist',
            options=[
                {'label': f"{file['filename']} (ID: {file['file_id']})", 'value': file['file_id']}
                for file in uploaded_files
            ],
            value=[]  # Initially, no files selected
        )
    ]

# Add a new callback for querying using selected files
@app.callback(
    Output('answer-output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('query-input', 'value'),
     State('file-checklist', 'value')]  # Capture selected files
)
def answer_question(n_clicks, query, selected_files):
    if n_clicks is None or query is None:
        return html.Div("Please submit a query.")

    if not selected_files:
        return html.Div("Please select at least one file for querying.")

    logger.info(f"Received query: {query}")
    logger.info(f"Selected files for query: {selected_files}")

    # Retrieve documents based on selected file IDs
    selected_documents = []
    for file_id in selected_files:
        file_data = file_collection.find_one({"file_id": file_id})
        if file_data:
            try:
                # Try reading the file as text, if it's a text file
                with open(file_data['path'], 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    selected_documents.append(Document(page_content=file_content, metadata={"source": file_data['filename']}))
            except UnicodeDecodeError:
                # Handle binary files like PDFs
                logger.warning(f"Failed to decode {file_data['filename']} as UTF-8. Trying binary mode.")
                with open(file_data['path'], 'rb') as file:
                    file_content = file.read()
                    selected_documents.append(Document(page_content=str(file_content), metadata={"source": file_data['filename']}))

    if not selected_documents:
        return html.Div("No valid documents found for the selected files.")

    # Ensure all selected documents are part of the retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": len(selected_documents)})

    # Create QA chain using all the selected documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Pass the query to the chain
    try:
        result = qa_chain({"query": query})
        return html.Div([
            html.H5("Answer:"),
            html.P(result['result']),
            html.H5("Sources:"),
            html.Ul([html.Li(f"Source: {doc.metadata['source']}") for doc in result['source_documents']])
        ])
    except Exception as e:
        logger.error(f"Failed to retrieve answer for query {query}: {e}")
        return html.Div(f"Error retrieving answer: {str(e)}")



if __name__ == '__main__':
    app.run_server(debug=True)
