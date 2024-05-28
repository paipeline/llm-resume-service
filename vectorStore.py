import os
import openai
import pinecone
from langchain_openai import OpenAIEmbeddings
from transformers import GPT2Tokenizer
import chromadb
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter

from langchain.embeddings import SentenceTransformerEmbeddings


class VectorStore:
    def __init__(self, openai_key_path, pinecone_key_path = "Placeholder, no need to add, will be translated to Chroma, api deprecated in the following update", pinecone_env="us-west1-gcp", index_name="book-chapters"):
        """
        This class is intended to be used for vector storage for user background similarity searches. The inputs will be the inference about a user's background,
        and this will map similar users' backgrounds in the embedded vector space. See details: https://medium.com/@eugene-s/the-rise-of-embedding-technology-and-vector-databases-in-ai-4a8db58eb332
        
        Args:
            openai_key_path (str): Path to the OpenAI API key.
            pinecone_key_path (str): Path to the Pinecone API key. Default is a placeholder value.
            pinecone_env (str): Pinecone environment name. Default is "us-west1-gcp".
            index_name (str): Name of the index to be used in Pinecone.
        
        Attributes:
            openai_client (openai.OpenAI): OpenAI client instance.
            pinecone_index (pinecone.Index): Pinecone index instance.
            namespace (str): Namespace for the vectors.
            tokenizer (GPT2Tokenizer): Tokenizer for text processing.
            embeddings (SentenceTransformerEmbeddings): Embedding function using SentenceTransformer.
            chroma_db (Chroma): Chroma database instance for storing and retrieving documents.
        """
        
        self.openai_key_path = openai_key_path
        self.openai_key = None
        self.index_name = index_name
        self.openai_client = self._load_openai()
        
        # init chroma
        self.namespace = None        
        self.chroma_db = self._initialize_chroma_db()

        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
        # self.embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # embeddings and chromaBD init
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")





    def _load_openai(self):
        """
        Load the OpenAI API key from the environment variable.

        Returns:
            client (openai.OpenAI): OpenAI client instance.

        Raises:
            ValueError: If the OpenAI API key environment variable is not set.
            Exception: For other errors during the loading process.
        """
        client = None
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except Exception as e:
            print(f"Error reading OpenAI API key: {e}")
        return client



    # def _initialize_pinecone(self):
    #     """
    #     Initialize the Pinecone index. [Will be deprecated in the following update.]

    #     Returns:
    #         index (pinecone.Index): Pinecone index instance.

    #     Raises:
    #         Exception: For errors during the Pinecone initialization process.
    #     """
    #     try:
    #         self.pinecone_key = os.getenv("PINECONE_API_KEY")
    #         pinecone.init(api_key=self.pinecone_key, environment="us-west1-gcp")
    #         if self.index_name not in pinecone.list_indexes():
    #             pinecone.create_index(self.index_name, dimension=384)
    #             print(f"-----Created and connected to index: {self.index_name}-----")
    #         else:
    #             print(f"-----Connected to index: {self.index_name}-----")
    #         return pinecone.Index(self.index_name)
    #     except Exception as e:
    #         print(f"An error occurred during Pinecone initialization: {e}")



    def _initialize_chroma_db(self):
        """
        Initialize the Chroma database.

        Returns:
            Chroma: Chroma database instance.

        Raises:
            Exception: For errors during the ChromaDB initialization process.
        """
        try:
            chroma_client = chromadb.Client()
            if (not self.collection_exists(chroma_client, self.namespace)):
                chroma_client.create_collection(self.namespace)
            else:
                print(f"Collection {self.namespace} already exists and ready to do operations.")
            return chroma_client

        except Exception as e:
            print(f"An error occurred during ChromaDB initialization: {e}")

    def collection_exists(client,collection_name):
        collections = client.list_collections()
        return any(collection["name"] == collection_name for collection in collections)


    def create_namespace(self, title):
        """
        Create a namespace based on the given title.

        Args:
            title (str): The title used to create the namespace.

        Returns:
            namespace (str): The created namespace.
        """
        self.namespace = title.replace(" ", "_").lower()
        print(f"Namespace created: {self.namespace}")
        return self.namespace




    def get_embedding(self, text):
        """
        Get the embedding of the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            embed (list): The embedding of the text.

        Raises:
            Exception: For errors during the embedding process.
        """
        
        try:
            embed = self.embeddings.embed_query(text)
            return embed
        except Exception as e:
            print(f"An error occurred while fetching the embedding: {e}")
            return None




            
    def upsert_inference(self, vector_id, text, override_mode=True):
        """
        Upsert the inference text into the ChromaDB.

        Args:
            vector_id (str): The ID of the vector.
            text (str): The text to be upserted.
            override_mode (bool): Whether to override the existing vector. Default is True.

        Raises:
            Exception: For errors during the upsert process.
        """
        
        if not override_mode and self._vector_exists(vector_id):
            print(f"Vector with ID {vector_id} already exists. Skipping upsert.")
            return
        # embedding = self.get_embedding(text)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        splits = text_splitter.split_text(text)
        docs = [Document(page_content=split, metadata={"id": vector_id}) for split in splits]
        try:
            self.chroma_db.from_documents(docs, self.embeddings)
            print(f"Safely upserted inference with ID {vector_id} in namespace {self.namespace}.")
        except Exception as e:
            print(f"Error upserting inference in ChromaDB: {e}")
            
           
            

    def _vector_exists(self, vector_id):
        """
        Check if a vector exists in Pinecone. [Will be deprecated in the following update.]

        Args:
            vector_id (str): The ID of the vector.

        Returns:
            bool: True if the vector exists, False otherwise.

        Raises:
            Exception: For errors during the vector existence check.
        """
        
        try:
            response = self.pinecone_index.fetch(ids=[vector_id], namespace=self.namespace)
            return vector_id in response['vectors']
        except Exception as e:
            print(f"An error occurred during vector existence check: {e}")
            return False




    def retrieve_embedding(self, vector_id):
        """
        Retrieve the embedding of a vector from Pinecone.

        Args:
            vector_id (str): The ID of the vector.

        Returns:
            vector (list or str): The retrieved vector or an error message if not found.

        Raises:
            Exception: For errors during the embedding retrieval process.
        """
        try:
            response = self.pinecone_index.fetch(ids=[vector_id], namespace=self.namespace)
            if response and "vectors" in response and vector_id in response["vectors"]:
                vector = response["vectors"][vector_id]["values"]
                return vector
            else:
                return "Draft not found"
        except Exception as e:
            print(f"An error occurred during embedding retrieval: {e}")
            return None





    def split_text(self, title, text, max_length=20):
        """
        Split the text into chunks based on the language (Chinese or English) and maximum length.

        Args:
            title (str): The title of the text.
            text (str): The text to be split.
            max_length (int): The maximum length of each chunk. Default is 20.

        Returns:
            list: A list of text chunks.
        """
        
        if self.is_chinese(text):
            return self.split_text_chinese(title, text, max_length)
        else:
            return self.split_text_english(title, text, max_length)

    def is_chinese(self, text):
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def split_text_chinese(self, title, text, max_length=20):
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_count = 1

        for char in text:
            if current_length + len(char) + 1 > max_length:
                chunk_title = f"{title}_{chunk_count}"
                chunks.append((chunk_title, "".join(current_chunk)))
                current_chunk = [char]
                current_length = len(char) + 1
                chunk_count += 1
            else:
                current_chunk.append(char)
                current_length += len(char) + 1

        if current_chunk:
            chunk_title = f"{title}_{chunk_count}"
            chunks.append((chunk_title, "".join(current_chunk)))

        return chunks




    def split_text_english(self, title, text, max_length=20):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_count = 1

        for word in words:
            if current_length + len(word) + 1 > max_length:
                chunk_title = f"{title}_{chunk_count}"
                chunks.append((chunk_title, " ".join(current_chunk)))
                current_chunk = [word]
                current_length = len(word) + 1
                chunk_count += 1
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunk_title = f"{title}_{chunk_count}"
            chunks.append((chunk_title, " ".join(current_chunk)))

        return chunks




    def calculate_tokens(self, text):
        """
        Calculate the number of tokens in the text using the GPT-2 tokenizer.

        Args:
            text (str): The text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        tokens = self.tokenizer.tokenize(text)
        return len(tokens)



    def calculate_tokens_chinese(self, text):
        """
        Calculate the number of tokens in the Chinese text.

        Args:
            text (str): The Chinese text to be tokenized.

        Returns:
            int: The number of tokens in the text.
        """
        tokens = list(text)
        return len(tokens)




    def insert_to_pinecone(self, text):
        """
        Insert the text into the Pinecone index. [Will be deprecated in the following update.]

        Args:
            text (dict): The text to be inserted, with 'title' and 'content' keys.

        Raises:
            Exception: For errors during the insertion process.
        """
        if len(text) == 1:
            self.create_namespace(text[0])
            return

        self.create_namespace(text['title'])

        chunks = self.split_text(text['title'], text['content'], max_length=1500)
        for chunk_title, chunk in chunks:
            metadata = text['context']
            text_with_metadata = f"{chunk}\n\nContext: {metadata}"
            self.upsert_embedding(chunk_title, text_with_metadata)
