from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from llama_index.embeddings.gemini import GeminiEmbedding
from pinecone import Pinecone
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex
import nest_asyncio
import os

nest_asyncio.apply()

load_dotenv()

parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], result_type="markdown", verbose=True)
file_extractor = {".pdf": parser}
#documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

llm=Groq(model="llama-3.1-70b-versatile",api_key=os.environ['GROQ_API_KEY'])

embedding_model=GeminiEmbedding(model="models/embedding-001",api_key=os.environ["GOOGLE_API_KEY"])

Settings.llm = llm
Settings.embed_model = embedding_model
Settings.chunk_size= 1024

pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index=pinecone_client.Index('project-1')
vector_store=PineconeVectorStore(pinecone_index=pinecone_index)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024,chunk_overlap=25),
        embedding_model
    ],
    vector_store=vector_store
)

#pipeline.run(documents=documents)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever=VectorIndexRetriever(index=index,similarity_top_k=5)
query_engine= RetrieverQueryEngine(retriever=retriever)

response = query_engine.query("what is this story about?")
print(response)