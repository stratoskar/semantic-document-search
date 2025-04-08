# Importing necessary libraries
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import pypdf
import docx

# Creating an instance of the Flask application.
app = Flask(__name__)

# Defining the folder where uploaded files will be stored.
UPLOAD_FOLDER = 'uploads'

# Checking if the 'uploads' folder exists. If it doesn't, it creates it.
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Creating a client to connect to the ChromaDB database.
chroma_client = chromadb.Client()

# Attempting to retrieve a collection named "my_collection" if it exists.
if "my_collection" in chroma_client.list_collections():
    collection = chroma_client.get_collection("my_collection")
else:
    collection = chroma_client.create_collection("my_collection")

# Loading a pre-trained Sentence Transformer model.
# This model will be used to convert the text of the files and the search query into embeddings (vector representations).
# Embeddings are numerical representations of the meaning of the text.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

# Function that takes text as input and returns its embedding using the loaded model.
def get_embedding(text):
    # The 'encode' method of the model converts the text into a numpy array (a table of numbers).
    # The 'tolist()' method converts this numpy array into a Python list.
    return model.encode(text).tolist()


# Function that calculates the cosine similarity between two vectors (embeddings).
# In the case of text embeddings, a higher cosine similarity value indicates greater semantic similarity.
def cosine_similarity(vec1, vec2):
    # Calculating the dot product of the two vectors.
    dot_product = np.dot(vec1, vec2)
    
    # Calculating the Euclidean norm (magnitude) of the first vector.
    norm_vec1 = np.linalg.norm(vec1)
    
    # Calculating the Euclidean norm (magnitude) of the second vector.
    norm_vec2 = np.linalg.norm(vec2)
    
    # Calculating the cosine similarity by dividing the dot product by the product of the norms.
    return dot_product / (norm_vec1 * norm_vec2)


# Function that reads the content of a file based on its extension.
def read_file_content(filepath):
    # If the file has a '.pdf' extension.
    if filepath.endswith('.pdf'):
        # Opens the file in binary read mode ('rb').
        with open(filepath, 'rb') as file:
            # Creates a PdfReader object to read the PDF.
            reader = pypdf.PdfReader(file)
            # Extracts the text from each page of the PDF and joins it into a single string.
            text = "".join(page.extract_text() for page in reader.pages)
        return text
    
    # If the file has a '.docx' extension.
    elif filepath.endswith('.docx'):
        # Opens the docx file.
        doc = docx.Document(filepath)
        # Extracts the text from each paragraph of the document and joins them with a newline character.
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text
    
    # If the file has a '.txt' extension.
    elif filepath.endswith('.txt'):
        # Opens the file in read mode ('r') with UTF-8 encoding to support various characters.
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
        
    # If the file extension is not one of the above, it returns None.
    else:
        return None


# When a user visits the root of the application with a GET request, the 'index' function is executed.
@app.route('/', methods=['GET'])
def index():
    # Returns the content of the 'index.html' file, which should be in the 'templates' folder (must exist).
    return render_template('index.html')


# Definition of the '/upload_search' route for handling file upload form submission (POST method).
@app.route('/upload_search', methods=['POST'])
def upload_search_file():
    # Checks if a file with the name 'file' is present in the request.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    # Retrieves the file from the request.
    file = request.files['file']
    
    # Checks if the filename is empty.
    if file.filename == '':
        # If empty, returns a JSON object with an error message.
        return jsonify({'error': 'No selected file'})
    
    # Checks if the file exists and if its filename ends with one of the supported extensions (.pdf, .docx, .txt), ignoring case.
    if file and file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
        
        # Creates the full path where the file will be saved on the server.
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        # Saves the uploaded file to the specified folder.
        file.save(filepath)
        # Reads the content of the file using the appropriate function based on the extension.
        content = read_file_content(filepath)
        # If the file content was read successfully.
        if content:
            # Creates the embedding for the content of the file.
            embedding = get_embedding(content)
            # Creates metadata for the file, in this case just the filename.
            metadata = {"filename": file.filename}

            # Adds the embedding, content, metadata, and a unique ID (the filename) to the ChromaDB collection.
            collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata],
                ids=[file.filename]
            )
            # Returns a JSON object with a success message.
            return jsonify({'message': 'File uploaded successfully'})
        # If there was an error reading the file content.
        else:
            return jsonify({'error': 'Error reading file content'})
    # If the file type is not supported.
    else:
        return jsonify({'error': 'Unsupported file type'})


# Definition of the '/search_similar' route for searching similar files (POST method).
@app.route('/search_similar', methods=['POST'])
def search_similar():
    # Checks if a file with the name 'file' is present in the request.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    # Checks if the filename is empty.
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Checks if the file exists and if its filename ends with one of the supported extensions (.pdf, .docx, .txt), ignoring case.
    if file and file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
        # Creates a temporary path to save the uploaded file.
        filepath = os.path.join(UPLOAD_FOLDER, "temp_" + file.filename)  # Temporary save
        # Saves the uploaded file temporarily.
        file.save(filepath)
        # Reads the content of the uploaded file for searching.
        content = read_file_content(filepath)
        # Deletes the temporary file.
        os.remove(filepath)  # Remove temp file

        # If the file content was read successfully.
        if content:
            # Creates the embedding for the search query text.
            query_embedding = get_embedding(content)
            # Performs a similarity search in the ChromaDB collection for the top 5 most similar documents.
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            # Initializes a list to store the similar files found.
            similar_files = []
            # Checks if there are results and if they contain IDs and distances.
            if results and results['ids'] and results['distances']:
                # Iterates through the results.
                for i, doc_id in enumerate(results['ids'][0]):
                    # Converts the distance to a similarity score (1 - distance).
                    similarity = 1 - results['distances'][0][i]
                    # If the similarity is above a certain threshold (0.1 in this case).
                    if similarity > 0.1:
                        # Appends a dictionary containing the filename and its similarity score to the list.
                        similar_files.append({
                            "filename": doc_id,
                            "similarity": similarity
                        })
            # Returns a JSON object with the search results.
            return jsonify({'results': similar_files})
        # If there was an error reading the content of the search file.
        else:
            return jsonify({'error': 'Error reading file content'})
    # If the file type for search is not supported.
    else:
        return jsonify({'error': 'Unsupported file type for search'})


# Definition of the '/all_files' route to retrieve the names of all stored files (GET method).
@app.route('/all_files', methods=['GET'])
def get_all_files():
    # Retrieves all entries from the collection, including their metadata.
    all_files = collection.get(include=["metadatas"])
    # Checks if there are files and if the metadata exists.
    if all_files and all_files['metadatas']:
        # Creates a list of filenames from the metadata.
        files = [meta['filename'] for meta in all_files['metadatas']]
        # Returns a JSON object with the list of filenames.
        return jsonify({'files': files})
    # If there are no files in the collection.
    else:
        return jsonify({'files': []})


# Definition of the '/download/<filename>' route to serve a file for download (GET method).
# '<filename>' is a placeholder that will be replaced with the actual filename in the URL.
@app.route('/download/<filename>')
def download_file(filename):
    # Uses Flask's 'send_from_directory' function to send the file from the 'UPLOAD_FOLDER'.
    # 'as_attachment=True' tells the browser to download the file instead of displaying it.
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


# This block of code is executed only when the script is run directly
if __name__ == '__main__':
    # Starts the Flask application
    app.run(debug=False)