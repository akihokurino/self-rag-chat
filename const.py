import os

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DB_HOST = os.getenv("DB_HOST", "localhost")
DATABASE_URL = f"postgresql+psycopg2://postgres:postgres@{DB_HOST}:5432/self_rag_chat"
EMBEDDING_MODEL = "text-embedding-3-small"
AZURE_DOCUMENT_INTELLIGENCE_URL = (
    "https://standard-plan-document-intelligence.cognitiveservices.azure.com/"
)
AZURE_DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
