import os

# Define the directory to search recursively
root_dir = "/home/shane/Desktop/llm_dashboard/anthropic_dashboard"  # Change this to the directory you want to search

# Define the output file where all the content will be saved
output_file = "collected_code.txt"

# Predefined comments to insert before and after each file's content
pre_comment = "/* Begin of the file: {filename} */\n"
post_comment = "/* End of the file: {filename} */\n"

long_term_goal = """

"""



# Beginning of text to insert before the file contents
message_intro = """
Project Overview:
This is source code for a dashboard to send queries to a LLM through API.  The dashboard with built with Plotly Dash.  It works for sending/receiving a query
to Claude. The dashboard is built with a modular design, with separate files for the dashboard layout, document manager, and RAG utilities.  The dashboard


Todo:
- Clearing documents from RAG does not seem to work completely as I still see "RAG Store: 1 documents, 94 chunks Direct Files: 7 files" even after deleting the files
- Add token count to the dashboard for each query
- Response from the LLM for this query seems to be very short.  is there a setting or something I'm missing?  
__________________________________________________________________________________________
Query:
tell me what documents you have access to and what they contain
__________________________________________________________________________________________
Response:

Selected Files:

    /home/shane/Desktop/llm_dashboard/anthropic_dashboard/uploads/20241025_075542_mcpc.pdf

Answer
Based on the provided context, I have access to two main types of documents: 1. A PDF document titled "20241025_075542_mcpc.pdf" which contains detailed documentation about the Motor Control Protocol Suite (MCPS) and Motor Control Protocol (MCP). This document covers: - Protocol specifications - Command structures - Register definitions - Communication procedures - Service descriptions (Command, Registry, Datalog, and Notification services) 2. A Python source code file "serialConnectionProtocol.py" which contains: - Implementation code for the protocol - Register definitions - Command implementations - Communication functions - CRC calculation utilities - Various helper functions for packet creation and data handling The PDF document appears to be the official protocol specification documentation, while the Python file is an implementation of that protocol. The documents contain detailed information about: - Protocol structure and services [Source: PDF Page 2] - Register definitions and types [Source: PDF Pages 27-33] - Command formats and responses [Source: PDF Pages 3-5] - Communication procedures including connection establishment [Source: PDF Pages 14-16] - Datalog and notification services [Source: PDF Pages 7-9] The documentation is focused on the communication protocol between a Controller device and a Performer device (typically an STM32 MCU running motor control applications) [Source: PDF Page 1]. This appears to be version 1 of the Motor Control Protocol as specified in the documentation [Source: PDF Page 2].
Sources

    20241025_075542_mcpc.pdf (Page 27)
    Relevance: 25.52%
    20241025_075542_mcpc.pdf (Page 7)
    Relevance: 24.76%
    20241025_075542_mcpc.pdf (Page 5)
    Relevance: 23.64%
    20241025_075542_mcpc.pdf (Page 5)
    Relevance: 22.88%
    20241025_075542_mcpc.pdf (Page 28)
    Relevance: 22.59%
    20241025_075542_mcpc.pdf (Page 9)
    Relevance: 22.44%
    20241025_075542_mcpc.pdf (Page 2)
    Relevance: 21.95%
    20241025_075542_mcpc.pdf (Page 6)
    Relevance: 21.74%
    20241025_075542_mcpc.pdf (Page 32)
    Relevance: 21.17%
    20241025_075542_mcpc.pdf (Page 31)
    Relevance: 20.79%
    20241025_075542_mcpc.pdf (Page 32)
    Relevance: 20.50%
    20241025_075542_mcpc.pdf (Page 2)
    Relevance: 20.30%
    20241025_075542_mcpc.pdf (Page 7)
    Relevance: 20.12%
    20241025_075542_mcpc.pdf (Page 6)
    Relevance: 19.55%
    20241025_075542_mcpc.pdf (Page 9)
    Relevance: 19.41%
    20241025_075542_mcpc.pdf (Page 25)
    Relevance: 19.26%
    20241025_075542_mcpc.pdf (Page 7)
    Relevance: 19.13%
    20241025_075542_mcpc.pdf (Page 8)
    Relevance: 19.07%
    20241025_075542_mcpc.pdf (Page 1)
    Relevance: 19.06%
    20241025_075542_mcpc.pdf (Page 33)
    Relevance: 19.04%
    20241025_075542_mcpc.pdf (Page 10)
    Relevance: 19.02%
    20241025_075542_mcpc.pdf (Page 15)
    Relevance: 18.86%
    20241025_075542_mcpc.pdf (Page 13)
    Relevance: 18.76%
    20241025_075542_mcpc.pdf (Page 14)
    Relevance: 18.65%
    20241025_075542_mcpc.pdf (Page 16)
    Relevance: 18.49%
    20241025_075542_mcpc.pdf (Page 31)
    Relevance: 18.43%
    20241025_075542_mcpc.pdf (Page 1)
    Relevance: 18.37%
    20241025_075542_mcpc.pdf (Page 8)
    Relevance: 18.24%
    20241025_075542_mcpc.pdf (Page 13)
    Relevance: 18.19%
    20241025_075542_mcpc.pdf (Page 28)
    Relevance: 17.65%
    20241025_075542_mcpc.pdf (Page 29)
    Relevance: 17.64%
    20241025_075542_mcpc.pdf (Page 8)
    Relevance: 17.34%
    20241025_075542_mcpc.pdf (Page 12)
    Relevance: 17.25%
    20241025_075542_mcpc.pdf (Page 8)
    Relevance: 17.13%
    20241025_075542_mcpc.pdf (Page 9)
    Relevance: 17.13%
    20241025_075542_mcpc.pdf (Page 10)
    Relevance: 16.92%
    20241025_075542_mcpc.pdf (Page 1)
    Relevance: 16.85%
    20241025_075542_mcpc.pdf (Page 1)
    Relevance: 16.80%
    20241025_075542_mcpc.pdf (Page 6)
    Relevance: 16.69%
    20241025_075542_mcpc.pdf (Page 25)
    Relevance: 16.66%
    20241025_075542_mcpc.pdf (Page 24)
    Relevance: 16.60%
    20241025_075542_mcpc.pdf (Page 16)
    Relevance: 16.15%
    20241025_075542_mcpc.pdf (Page 24)
    Relevance: 16.02%
    20241025_075542_mcpc.pdf (Page 22)
    Relevance: 16.00%
    20241025_075542_mcpc.pdf (Page 2)
    Relevance: 16.00%
    20241025_075542_mcpc.pdf (Page 15)
    Relevance: 15.95%
    20241025_075542_mcpc.pdf (Page 15)
    Relevance: 15.83%
    20241025_075542_mcpc.pdf (Page 4)
    Relevance: 15.82%
    20241025_075542_mcpc.pdf (Page 30)
    Relevance: 15.47%
    20241025_075542_mcpc.pdf (Page 16)
    Relevance: 15.39%
    20241025_075542_mcpc.pdf (Page 30)
    Relevance: 15.26%
    20241025_075542_mcpc.pdf (Page 3)
    Relevance: 15.15%
    20241025_075542_mcpc.pdf (Page 14)
    Relevance: 15.14%
    20241025_075542_mcpc.pdf (Page 9)
    Relevance: 15.14%
    20241025_075542_mcpc.pdf (Page 16)
    Relevance: 15.10%
    20241025_075542_mcpc.pdf (Page 3)
    Relevance: 14.88%
    20241025_075542_mcpc.pdf (Page 22)
    Relevance: 14.85%
    20241025_075542_mcpc.pdf (Page 11)
    Relevance: 14.52%
    20241025_075542_mcpc.pdf (Page 2)
    Relevance: 14.49%
    20241025_075542_mcpc.pdf (Page 28)
    Relevance: 14.41%
    20241025_075542_mcpc.pdf (Page 10)
    Relevance: 14.40%
    20241025_075542_mcpc.pdf (Page 29)
    Relevance: 14.02%
    20241025_075542_mcpc.pdf (Page 8)
    Relevance: 13.80%
    20241025_075542_mcpc.pdf (Page 5)
    Relevance: 13.64%
    20241025_075542_mcpc.pdf (Page 3)
    Relevance: 13.63%
    20241025_075542_mcpc.pdf (Page 4)
    Relevance: 13.56%
    20241025_075542_mcpc.pdf (Page 32)
    Relevance: 13.40%
    20241025_075542_mcpc.pdf (Page 27)
    Relevance: 13.17%
    20241025_075542_mcpc.pdf (Page 30)
    Relevance: 12.86%
    20241025_075542_mcpc.pdf (Page 12)
    Relevance: 12.80%
    20241025_075542_mcpc.pdf (Page 27)
    Relevance: 12.43%
    20241025_075542_mcpc.pdf (Page 14)
    Relevance: 12.39%
    20241025_075542_mcpc.pdf (Page 10)
    Relevance: 12.38%
    20241025_075542_mcpc.pdf (Page 11)
    Relevance: 12.37%
    20241025_075542_mcpc.pdf (Page 7)
    Relevance: 12.30%
    20241025_075542_mcpc.pdf (Page 24)
    Relevance: 12.00%
    20241025_075542_mcpc.pdf (Page 22)
    Relevance: 11.73%
    20241025_075542_mcpc.pdf (Page 26)
    Relevance: 10.95%
    20241025_075542_mcpc.pdf (Page 5)
    Relevance: 10.68%
    20241025_075542_mcpc.pdf (Page 11)
    Relevance: 10.17%
    20241025_075542_mcpc.pdf (Page 14)
    Relevance: 10.11%
    20241025_075542_mcpc.pdf (Page 11)
    Relevance: 9.67%
    20241025_075542_mcpc.pdf (Page 23)
    Relevance: 9.56%
    20241025_075542_mcpc.pdf (Page 26)
    Relevance: 9.14%
    20241025_075542_mcpc.pdf (Page 29)
    Relevance: 9.01%
    20241025_075542_mcpc.pdf (Page 15)
    Relevance: 8.78%
    20241025_075542_mcpc.pdf (Page 13)
    Relevance: 8.13%
    20241025_075542_mcpc.pdf (Page 10)
    Relevance: 7.93%
    20241025_075542_mcpc.pdf (Page 23)
    Relevance: 7.83%
    20241025_075542_mcpc.pdf (Page 12)
    Relevance: 7.73%
    20241025_075542_mcpc.pdf (Page 23)
    Relevance: 6.75%
    20241025_075542_mcpc.pdf (Page 22)
    Relevance: 6.73%
    20241025_075542_mcpc.pdf (Page 23)
    Relevance: 5.38%
    20241025_075542_mcpc.pdf (Page 24)
    Relevance: 2.30%

Query Statistics

Context chunks used: 95

Unique sources: 1
__________________________________________________________________________________________
Terminal output:
(venv) shane@shane-ThinkPad-X1-Carbon-Gen-10:~/Desktop/llm_dashboard/anthropic_dashboard$ python3 app.py
INFO:rag_utils:RAG Manager initialized with path: /home/shane/Desktop/llm_dashboard/anthropic_dashboard/vectorstore
INFO:chromadb.telemetry.product.posthog:Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
INFO:rag_utils:All components initialized successfully
INFO:dashboard_layout:Dashboard initialized successfully
INFO:dashboard_layout:Starting dashboard on port 8050
Dash is running on http://127.0.0.1:8050/

INFO:dash.dash:Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'dashboard_layout'
 * Debug mode: on
INFO:rag_utils:RAG Manager initialized with path: /home/shane/Desktop/llm_dashboard/anthropic_dashboard/vectorstore
INFO:chromadb.telemetry.product.posthog:Anonymized telemetry enabled. See                     https://docs.trychroma.com/telemetry for more information.
INFO:rag_utils:All components initialized successfully
INFO:dashboard_layout:Dashboard initialized successfully
INFO:dashboard_layout:Starting dashboard on port 8050
INFO:rag_utils:Clearing vector store
INFO:rag_utils:Vector store cleared and reinitialized successfully
INFO:document_manager:Successfully saved file: mcpc.pdf (rag)
INFO:rag_utils:Processing file: mcpc.pdf (/home/shane/Desktop/llm_dashboard/anthropic_dashboard/uploads/20241025_075542_mcpc.pdf)
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_0
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_1
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_2
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_3
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_4
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_5
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_6
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_7
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_8
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_9
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_10
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_11
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_12
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_13
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_14
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_15
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_16
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_17
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_18
WARNING:chromadb.segment.impl.metadata.sqlite:Insert of existing embedding ID: 20241025_075544_mcpc.pdf_19
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_0
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_1
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_2
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_3
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_4
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_5
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_6
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_7
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_8
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_9
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_10
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_11
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_12
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_13
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_14
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_15
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_16
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_17
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_18
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Add of existing embedding ID: 20241025_075544_mcpc.pdf_19
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
INFO:rag_utils:Successfully processed mcpc.pdf into 114 chunks
INFO:document_manager:Successfully saved file: serialConnectionProtocol.py (prompt)
INFO:document_manager:Successfully loaded prompt file: serialConnectionProtocol.py
INFO:rag_utils:Querying with: please tell me what documents you have access to and what is their contents
INFO:rag_utils:Selected RAG files: []
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Number of requested results 1000 is greater than number of elements in index 94, updating n_results = 94
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO:rag_utils:Querying with: can you now see the python file?
INFO:rag_utils:Selected RAG files: []
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Number of requested results 1000 is greater than number of elements in index 94, updating n_results = 94
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
INFO:document_manager:Successfully saved file: serialConnectionProtocol.py (prompt)
INFO:document_manager:Successfully loaded prompt file: serialConnectionProtocol.py
INFO:rag_utils:Querying with: tell me what documents you have access to and what they contain
INFO:rag_utils:Selected RAG files: ['/home/shane/Desktop/llm_dashboard/anthropic_dashboard/uploads/20241025_075542_mcpc.pdf']
INFO:httpx:HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
WARNING:chromadb.segment.impl.vector.local_persistent_hnsw:Number of requested results 1000 is greater than number of elements in index 94, updating n_results = 94
INFO:httpx:HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"

"""


# Define the file extensions to search for
file_extensions = ['.py']

# List to store the collected files
collected_files = []

# Function to check if a file has the correct extension
def has_valid_extension(filename):
    return any(filename.endswith(ext) for ext in file_extensions)

# Define the specific filenames to search for
specific_filenames = ['app.py',  'document_manager.py', 'rag_utils.py', 'dashboard_layout.py']

# Function to check if a file has one of the specific filenames
def has_specific_filename(filename):
    return os.path.basename(filename) in specific_filenames

collected_files = []

# Get all files in the root directory (non-recursively)
for file in os.listdir(root_dir):
    if file in specific_filenames:
        file_path = os.path.join(root_dir, file)
        # Make sure it's a file and not a directory
        if os.path.isfile(file_path):
            collected_files.append(file_path)

# Sort the files to ensure consistent output
collected_files.sort()

# Open the output file for writing the collected content
with open(output_file, 'w') as out_file:
    out_file.write(message_intro)
    
    for file_path in collected_files:
        try:
            # Read the content of the file
            with open(file_path, 'r') as file:
                content = file.read()

            # Write the predefined comments and the file content to the output file
            relative_path = os.path.basename(file_path)  # Only show filename instead of full path
            out_file.write(pre_comment.format(filename=relative_path))
            out_file.write(content)
            out_file.write("\n" + post_comment.format(filename=relative_path) + "\n")

        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

print(f"Collected {len(collected_files)} files and saved to {output_file}.")