from src.utils import inicio_processo, fim_processo
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance, models


# Chunks DOCUMENTS
def get_chunks_documents(documents_for_chunks: list[Document]) -> list[Document]:
    documents_splitter = CharacterTextSplitter(separator='\n',
                                               chunk_size=1000,
                                               chunk_overlap=100,
                                               length_function=len)

    chunks: list[Document] = documents_splitter.split_documents(documents=documents_for_chunks)

    return chunks


# Chunks TEXTS
def get_chunks_text(text_for_chunks) -> list[str]:
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=1000,
                                          chunk_overlap=100,
                                          length_function=len)

    chunks: list[str] = text_splitter.split_text(text=text_for_chunks)

    return chunks


if __name__ == '__main__':
    # Data e hora do INÍCIO do processo.
    dataHoraInicioProcesso: datetime = inicio_processo()

    colletion_name = 'documentos_claro_geodata_llama2_7b'

    # loading .env configurations file - ()
    load_dotenv(r'D:\Users\xxxxx\Desenvolvimentos\ProjetosPython\ProjetosAutonomos\pjtLangChain\.env')

    # Loader PDF document - (Carrega documentos PDF)
    loader = PyPDFLoader(file_path=os.getenv('LOCAL_PDF_FILE').replace('\\', '/') + os.getenv('NAME_PDF_FILE'))

    # Creates a list with parts of the document to embed
    documents: list[Document] = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(separators=['\n'],
                                                                                                   chunk_size=1000,
                                                                                                   chunk_overlap=100,
                                                                                                   length_function=len,
                                                                                                   is_separator_regex=False))

    # Configure the client - (Configura cliente)
    client_qdrant: QdrantClient = QdrantClient(location='localhost',
                                               port=6333)

    # Create Collections - (Criar Coleções)
    if not client_qdrant.collection_exists(colletion_name):
        client_qdrant.create_collection(collection_name=colletion_name,
                                        vectors_config=models.VectorParams(size=4096,
                                                                           distance=Distance.COSINE),
                                        on_disk_payload=True)

    # Embeddings - (Incorporações)
    embedding_ollama = OllamaEmbeddings(model='llama2:7b',
                                        temperature=0.9)

    # Create a vector store (QDrant) - (Crie um armazenamento de vetores (QDrant))
    vs_qdrant: Qdrant = Qdrant(client=client_qdrant,
                               collection_name=colletion_name,
                               embeddings=embedding_ollama)

    # Persists ducuments in vector database - (Persiste documentos no banco de dados vetorial)
    vs_qdrant.add_documents(documents=documents)

    print('Total of persisted document(s): {0}'.format(len(documents)))

    # Lista
    points_teste_collection = client_qdrant.get_collection(collection_name=colletion_name)

    print(points_teste_collection)

    # Data e hora do FIM do processo.
    fim_processo(dataHoraInicioProcesso=dataHoraInicioProcesso)
