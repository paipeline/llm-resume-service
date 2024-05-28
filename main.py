from multiprocessing import set_start_method, get_start_method

# Set the start method for multiprocessing 
# FIXED MEMORY PROBLEM: ADDED TO SOLVE RELATED ISSUES OF MEMORY
if __name__ == "__main__":
    try:
        if get_start_method(allow_none=True) is None:
            set_start_method('spawn')
    except RuntimeError:
        pass

from flasgger import Swagger
from flask import Flask, jsonify, request
from utils import extract_and_infer, retrieve_top_documents
from vectorStore import VectorStore



app = Flask(__name__)
swagger = Swagger(app)

@app.route("/")
def index():
    return jsonify({
        "message": "Welcome to the Resume Processing API",
        "endpoints": {
            "/resume/upload": "Upload a PDF file and generate a resume summary",
            "/documents/retrieve": "Retrieve the top documents that best fit the provided query"
        }
    })



@app.route("/resume/upload", methods=["POST"])
def process_resume():
    """
    Upload a PDF file and generate a resume summary
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
    responses:
      200:
        description: Inference of the resume
        schema:
          id: ResumeSummary
          properties:
            extracted_info:
              type: object
              description: The extracted information from the resume
            summary:
              type: object
              description: The inference based on the extracted info of the resume
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        #save the uploaded resume to a default location
        file_path = "./resume/resume_example.pdf"
        file.save(file_path) 
        
        #extract and infer information from the uploaded resume
        extracted_info, summary = extract_and_infer(file_path)
        
        response = {
            "extracted_info": extracted_info,
            "Inference": summary
        }
        return jsonify(response), 200

@app.route("/documents/retrieve", methods=["POST"])
def retrieve_documents():
    """
    Retrieve the top documents that best fit the provided query
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: query
        description: The query to find the best fit documents
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              description: The query text
    responses:
      200:
        description: Top documents that best fit the query
        schema:
          id: TopDocuments
          properties:
            documents:
              type: array
              items:
                type: object
                properties:
                  score:
                    type: number
                    description: Similarity score
                  extracted_info:
                    type: object
                    description: Extracted information of the document
    """
    data = request.get_json()
    
    # add inference of the selected user as query
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Retrieve the top documents that best fit the query
    top_matches = retrieve_top_documents(query = query, top_k= 10)
    
    response = {
        "documents": top_matches
    }
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)
