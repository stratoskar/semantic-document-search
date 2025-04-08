# Semantic Document Search Application

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-%23000000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-lightgrey?style=flat&logo=chromadb)](https://www.trychroma.com/)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-white?style=flat&logo=huggingface)](https://www.sbert.net/)

This web application allows users to upload text-based documents (PDF, DOCX, TXT) and then search for documents with semantically similar content by uploading another query document. It leverages Sentence Transformers for generating text embeddings and ChromaDB for efficient vector storage and similarity search.

## Overview

The application provides the following functionalities:

* **File Upload:** Users can upload documents in PDF, DOCX, or TXT format.
* **Semantic Search:** Users can upload a query document, and the application will find and display a list of previously uploaded documents that are semantically similar based on their content.
* **Document Storage:** Uploaded files are stored in the `uploads` directory on the server.
* **Vector Database:** The semantic embeddings of the uploaded documents are stored and indexed in a ChromaDB vector database for fast similarity searching.
* **Download:** Users can download any of the previously uploaded files.
* **List All Files:** Users can view a list of all uploaded files.

## Technologies Used

* **Python:** The primary programming language.
* **Flask:** A micro web framework for building the web application.
* **ChromaDB:** A vector database for storing and querying text embeddings.
* **Sentence Transformers:** A Python library for generating high-quality text embeddings.
* **PyPDF:** A library for reading PDF files.
* **python-docx:** A library for reading Microsoft Word (.docx) files.
* **Pillow (PIL):** A library for image processing (although currently not the core functionality).
* **NumPy:** A library for numerical operations, used in cosine similarity calculation.

## Setup and Installation

1.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask application:**
    ```bash
    python app.py
    ```

## Usage

1.  **Upload Documents:** On the main page, you can upload PDF, DOCX, or TXT files. These files will be processed and their semantic embeddings will be stored.
2.  **Search for Similar Documents:** Upload a query document (PDF, DOCX, or TXT). The application will display a list of the most semantically similar documents from the uploaded collection, along with their similarity scores.
