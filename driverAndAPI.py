from typing import List, Tuple
import langchain
import chromadb

# LangChain and Chromadb is used as a base from existing technologies so I put the definitions in the classes themselves

class LegalQAAPI:
    """
    API for Question Answering (QA) System for Legal Documents using LangChain.

    This API provides functionalities to upload legal documents, ask questions based on these documents, 
    and retrieve relevant reference documents.

    LangChain is a toolkit for developing language and legal models, providing
    functionalities for processing legal documents, generating answers to user queries,
    and retrieving relevant information.

    ChromaDB is an efficient database designed for storing and querying high-dimensional vector data, 
    optimized for tasks like similarity search in NLP applications.

    """

    def __init__(self, model_type: str, model_path: str, model_n_ctx: int, persist_directory: str):
        """
        Initialize the LegalQAAPI with LangChain components.

        Parameters:
            model_type (str): Type of the language model to be used.
            model_path (str): Path to the language model.
            model_n_ctx (int): Maximum number of tokens to consider for context.
            persist_directory (str): Directory to persist embeddings.
        """
        # self.embeddings_model_name = "your_embeddings_model_name"
        # self.persist_directory = persist_directory

        # # Initialize LangChain components
        # self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
        # self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

        # # Initialize LangChain retriever and LLM
        # self.retriever = self.db.as_retriever(search_kwargs={"k": 4})
        # self.llm = RetrievalQA.from_chain_type(model_type=model_type, model_path=model_path, max_tokens=model_n_ctx, retriever=self.retriever)

    def upload_documents(self, legal_documents: List[str]) -> str:
        """
        Upload legal documents to the system.

        Parameters:
            legal_documents (List[str]): List of legal documents in text format.

        Returns:
            str: Status message indicating whether the documents were successfully uploaded.
        """

        self.db.index_documents(legal_documents) # Pass the documents to LangChain for processing
        return "Documents uploaded successfully."

    def ask_question(self, user_query: str) -> Tuple[str, List[str]]:
        """
        Answer user's query based on uploaded legal documents.

        Parameters:
            user_query (str): User's query to be answered.

        Returns:
            Tuple[str, List[str]]: Answer to the user's query and relevant reference documents.
        """
        answer, reference_documents = self.llm.answer(user_query) # Extract the answer and relevant documents using LangChain
        return answer, reference_documents

class LegalQADriver:
    """
    Driver program for the Question Answering (QA) System for Legal Documents using LangChain.

    Orchestrates the internal workflow of the QA system.

    This driver program provides methods to collect and preprocess legal documents, 
    retrieve reference documents, answer user's questions, and evaluate the system's performance.

    LangChain is a toolkit for developing language and legal models, providing
    functionalities for processing legal documents, generating answers to user queries,
    and retrieving relevant information.

    ChromaDB is an efficient database designed for storing and querying high-dimensional vector data, 
    optimized for tasks like similarity search in NLP applications.
    """

    def __init__(self, model_type: str, model_path: str, model_n_ctx: int, persist_directory: str):
        """
        Initialize the LegalQADriver with LangChain components.

        Parameters:
            model_type (str): Type of the language model to be used.
            model_path (str): Path to the language model.
            model_n_ctx (int): Maximum number of tokens to consider for context.
            persist_directory (str): Directory to persist embeddings.
        """
        # self.persist_directory = persist_directory

        # # Initialize LangChain components
        # self.embeddings_model_name = "your_embeddings_model_name"
        # self.persist_directory = persist_directory
        # self.embeddings = HuggingFaceEmbeddings(model_name=self.embeddings_model_name)
        # self.db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

        # # Initialize LangChain retriever and LLM
        # self.retriever = self.db.as_retriever(search_kwargs={"k": 4})
        # self.llm = RetrievalQA.from_chain_type(model_type=model_type, model_path=model_path, max_tokens=model_n_ctx, retriever=self.retriever)

    def collect_and_preprocess_documents(self, legal_documents: List[str]) -> List[str]:
        """
        Collect and preprocess legal documents.

        Parameters:
            legal_documents (List[str]): List of legal documents in text format.

        Returns:
            List[str]: Preprocessed legal documents.
        """
        # Pass the documents to LangChain for preprocessing
        self.db.index_documents(legal_documents)
        return legal_documents

    def retrieve_reference_documents(self, user_query: str) -> List[str]:
        """
        Retrieve relevant reference documents based on user's query.

        Parameters:
            user_query (str): User's query to retrieve reference documents for.

        Returns:
            List[str]: Relevant reference documents.
        """
        # Use LangChain to retrieve relevant reference documents
        reference_documents = self.llm.relevant_documents(user_query)
        return reference_documents

    def answer_question(self, user_query: str) -> str:
        """
        Answer user's query based on uploaded legal documents.

        Parameters:
            user_query (str): User's query to be answered.

        Returns:
            str: Answer to the user's query.
        """
        # Get the answer using LangChain
        answer, _ = self.llm.answer(user_query)
        return answer

    def evaluate_system(self, test_questions: List[str], expected_answers: List[str]) -> float:
        """
        Evaluate the QA system's performance.

        Parameters:
            test_questions (List[str]): List of test questions.
            expected_answers (List[str]): List of expected answers corresponding to the test questions.

        Returns:
            float: Evaluation score (e.g., F1-score, accuracy).
        """
        evaluation_score = 0.62  #  Random value for now, would get a score based on some methadology
        return evaluation_score


# Below is the expected input -> function call -> output flow you would follow to obtain a answer
############################################################################################################
# the model you would pass in and the path to it, the amount of inputs you would pass in , etc (paramters)
api = LegalQAAPI(model_type="GPT4All", model_path="path/to/model", model_n_ctx=512, persist_directory="path/to/persist_directory")

# the legal documents you would read in
legal_documents = [
    "This is the first legal document.",
    "This is the second legal document.",
    "This is the third legal document."
]

api.upload_documents(legal_documents) # uploading the legal documents

user_query = "When did they mention about law 25" # User inputs a random query

answer, reference_documents = api.ask_question(user_query)

# Printing the results
print("Answer:", answer) 
print("Reference Documents:", reference_documents)
