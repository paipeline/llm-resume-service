import os
import pytesseract
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import openai
from pdf2image import convert_from_path
import json
from sentence_transformers import SentenceTransformer
from vectorStore import VectorStore 
from datetime import datetime

import chromadb
from dotenv import load_dotenv
            
storage_path = './chroma_db/'
print(storage_path)
load_dotenv('.env.local')
if storage_path is None:
    raise ValueError('STORAGE_PATH environment variable is not set')
client = chromadb.PersistentClient(path= storage_path)
collection = client.get_or_create_collection(name="test")



# init sentence transformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# init vectorStore
# vector_store = VectorStore(openai_key_path="path/to/openai_key.txt")

openai_api_key = os.environ["OPENAI_API_KEY"]
if not openai_api_key:
    raise ValueError("Did not find openai_api_key, please add an environment variable `OPENAI_API_KEY` which contains it, or pass `openai_api_key` as a named parameter.")

openai.api_key = openai_api_key


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using OCR.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    images = convert_from_path(pdf_path)
    resume_content = ""
    for image in images:
        resume_content += pytesseract.image_to_string(image)
    return resume_content




def extract_personal_info(resume_text):
    """
    Extract personal information from the resume text using a language model.

    Args:
        resume_text (str): Text content of the resume.

    Returns:
        str: Extracted personal information in JSON format.
    """
    
    prompt = PromptTemplate(
        template="""
        Extract the personal information from the following resume text in JSON format:
        Resume text:
        {resume_text}
        Personal Information:
        {{
            "name": "",
            "address": "",
            "email": "",
            "telephone number(optional)" : "",
            "awards" : [""], // any awards or certifications, add more if needed
        }}
        """,
        input_variables=["resume_text"],
    )
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"resume_text": resume_text})
    return result





def extract_education(resume_text):
    """
    Extract education details from the resume text using a language model.

    Args:
        resume_text (str): Text content of the resume.

    Returns:
        str: Extracted education details in JSON format.
    """
    
    prompt = PromptTemplate(
        template="""
        Extract the education details from the following resume text in JSON format:
        Resume text:
        {resume_text}
        Education:
        [
            {{
                "school": "",
                "degree": "",
                "graduation_year": "",
                "gpa/grade (optional)": None / ""                
            }},
            // add more as needed 
        ]
        """,
        input_variables=["resume_text"],
    )
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"resume_text": resume_text})
    return result




def extract_work_experience(resume_text):
    """
    Extract work experience details from the resume text using a language model.

    Args:
        resume_text (str): Text content of the resume.

    Returns:
        str: Extracted work experience details in JSON format.
    """
    prompt = PromptTemplate(
        template="""
        Extract the work experience details from the following resume text in JSON format:
        Resume text:
        {resume_text}
        Work Experience:
        [
            {{
                "company": "", // must be an eligible company name
                "position": "",
                "duration": "",
                "skills involved": "", // inference information from the work description
            }},
            // add more as needed
        ]
        """,
        input_variables=["resume_text"],
    )
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"resume_text": resume_text})
    return result





def extract_projects_and_skills(resume_text):
    """
    Extract project experience and skills from the resume text using a language model.

    Args:
        resume_text (str): Text content of the resume.

    Returns:
        str: Extracted project experience and skills in JSON format.
    """
    prompt = PromptTemplate(
        template="""
        Extract the project experience and skills from the following resume text in JSON format:
        Resume text:
        {resume_text}
        Project Experience:
        [
            {{
                "name": "",
                "duration": "",
                "role": "",
                "technologies_used": "",
                "description": "",
                "achievements": "",
                "team_size": "",
                "responsibilities": ""
            }},
            //add more as needed
        ],
        
        Skills:
        [
            "", // technical skills and soft skills, inferred if possible
            // add more as needed
        ]
        """,
        input_variables=["resume_text"],
    )
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"resume_text": resume_text})
    print(result)
    return result





def generate_inference(extracted_info):
    """
    Generate a summary of the resume based on extracted information using a language model.

    Args:
        extracted_info (str): Extracted information from the resume in JSON format.

    Returns:
        str: Generated summary of the resume in JSON format.
    """
    
    
    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)

    prompt = PromptTemplate(
        template="""
        You are a critical hiring manager. Based on the following extracted resume information, generate a detailed critical (positive and negative aspects) inference about the candidate's background information in the corresponding language of the candidate in JSON format:
        Extracted Information:
        {extracted_info}
        Add an inference of the candidate based on their strengths, skills, and experience.
        Output format:
        {{
            "inference" :"", // Analyzing the time, location, work experience, projects and awards, make a detailed deep analysis paragraph about candidate's potential career based on the extracted information.

        }}
        """,
        input_variables=["extracted_info"],
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    inference = chain.run({"extracted_info": extracted_info})
    
    return inference



def extract_and_infer(pdf_path):
    """
    Extract text from a PDF, extract information from the text, generate a summary, and store embeddings in ChromaDB.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        tuple: Extracted information and generated summary.
    """
    
    resume_text = extract_text_from_pdf(pdf_path)
    
    personal_info = extract_personal_info(resume_text)
    education = extract_education(resume_text)
    work_experience = extract_work_experience(resume_text)
    projects_and_skills = extract_projects_and_skills(resume_text)
    
    # convert to dict
    extracted_info_json = {
        "personal_information": json.loads(personal_info),
        "education": json.loads(education),
        "work_experience": json.loads(work_experience),
        "projects_and_skills": json.loads(projects_and_skills),
    }
    
    inference = generate_inference(json.dumps(extracted_info_json))
    inference_json = json.loads(inference)

    # Collecting data for the collection
    inferences = inference_json.get('inference')
    name = extracted_info_json.get('personal_information').get('name', 'Unknown Name')
    
    skills = extracted_info_json.get('projects_and_skills').get('Skills', [])
    current_timestamp = datetime.now().isoformat()
    metadatas = [
        {"source": "inference", "timestamp": current_timestamp, "author": "admin_test"},
    ]
    
    # Adding the documents to the collection
    print("Inferences:", inferences)
    print("Skills:", skills)
    print("Metadatas:", metadatas)
    print("IDs:", name)
    
    collection.upsert(
        documents=[inferences],
        metadatas=metadatas,
        ids=[name]
    )
    print("Background correctly added to the collection.")

    return extracted_info_json, inference_json





def retrieve_top_documents(query = "", top_k=5): # query should be the inference of the current selected user 
    # example query: "The candidate has a strong background in software engineering and has worked on multiple projects using Python and Java."
    """
    Retrieve the top documents that best fit the provided query using embeddings.

    Args:
        query (str): Query text.
        top_k (int): Number of top documents to retrieve. Default is 5.

    Returns:
        list: List of top documents with their scores, extracted information, and summaries.
    """

    #generate embedding for the query
    # query_embedding = model.encode(query).tolist()
    
    # # Search
    # results = vector_store.pinecone_index.query(
    #     vector=query_embedding,
    #     top_k=top_k,
    #     include_metadata=True,
    #     namespace=vector_store.namespace
    # )
    
    # top_documents = []
    # for match in results['matches']:
    #     top_documents.append({
    #         "score": match['score'],
    #         "extracted_info": match['metadata'],
    #         "summary": match['metadata'].get('summary')
    #     })
    
    res = collection.query(
        query_texts=query,
        n_results=top_k,
    )
    
    
    
    return res
