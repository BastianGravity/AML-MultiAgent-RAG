"""
Document processing pipeline endpoints for AML documents.

Provides endpoints for:
1. Processing PDFs (text extraction)
2. Chunking documents
3. Generating embeddings
4. Storing embeddings in Qdrant
"""

import os
import json
import logging
from fastapi import APIRouter
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process-pdfs")
async def process_pdfs():
    """
    Process PDF files from input_docs/ directory.
    
    Extracts text and metadata from all PDFs.
    """
    try:
        from backend.services.document_processing.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        documents = processor.process_all_pdfs()
        
        # Save processed documents
        os.makedirs("input_docs/processed", exist_ok=True)
        with open("input_docs/processed/processed_docs.json", "w") as f:
            json.dump(documents, f, indent=2)
        
        logger.info(f"Processed {len(documents)} documents")
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "message": f"Successfully processed {len(documents)} documents"
        }
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/chunk-documents")
async def chunk_documents():
    """
    Chunk processed documents into smaller segments.
    
    Creates overlapping chunks for embedding generation.
    """
    try:
        from backend.services.document_processing.text_splitter import TextChunker
        
        # Load processed documents
        processed_path = Path("input_docs/processed/processed_docs.json")
        if not processed_path.exists():
            return {
                "status": "error",
                "error": "No processed documents found. Run process-pdfs first."
            }
        
        with open(processed_path, "r") as f:
            documents = json.load(f)
        
        chunker = TextChunker()
        chunked_docs = chunker.chunk_documents(documents)
        
        # Save chunked documents
        os.makedirs("input_docs/processed", exist_ok=True)
        with open("input_docs/processed/chunked_docs.json", "w") as f:
            json.dump(chunked_docs, f, indent=2)
        
        logger.info(f"Created {len(chunked_docs)} chunks")
        
        return {
            "status": "success",
            "chunks_created": len(chunked_docs),
            "message": f"Successfully created {len(chunked_docs)} chunks"
        }
    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/generate-embeddings")
async def generate_embeddings():
    """
    Generate embeddings for chunked documents.
    
    Uses OpenAI or local embeddings depending on availability.
    """
    try:
        from backend.services.embeddings.openai_embeddings import OpenAIEmbeddings
        
        # Load chunked documents
        chunked_path = Path("input_docs/processed/chunked_docs.json")
        if not chunked_path.exists():
            return {
                "status": "error",
                "error": "No chunked documents found. Run chunk-documents first."
            }
        
        with open(chunked_path, "r") as f:
            chunked_docs = json.load(f)
        
        embeddings_service = OpenAIEmbeddings()
        embedded_docs = embeddings_service.embed_documents(chunked_docs)
        
        # Save embedded documents
        os.makedirs("input_docs/embeddings", exist_ok=True)
        with open("input_docs/embeddings/embedded_docs.json", "w") as f:
            json.dump(embedded_docs, f, indent=2)
        
        logger.info(f"Generated {len(embedded_docs)} embeddings")
        
        return {
            "status": "success",
            "embeddings_created": len(embedded_docs),
            "message": f"Successfully generated {len(embedded_docs)} embeddings"
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/store-embeddings")
async def store_embeddings():
    """
    Store embeddings in Qdrant vector database.
    
    Uploads all embeddings to the configured collection.
    """
    try:
        from backend.services.vector_db.qdrant_client import QdrantVectorDB
        
        embedded_docs_path = Path("input_docs/embeddings/embedded_docs.json")
        if not embedded_docs_path.exists():
            return {
                "status": "error",
                "error": "No embedded documents found. Run generate-embeddings first."
            }
        
        with open(embedded_docs_path, "r") as f:
            embedded_docs = json.load(f)
        
        vector_db = QdrantVectorDB()
        success = vector_db.store_embeddings(embedded_docs)
        
        if success:
            logger.info(f"Stored {len(embedded_docs)} embeddings in Qdrant")
            return {
                "status": "success",
                "embeddings_stored": len(embedded_docs),
                "message": f"Successfully stored {len(embedded_docs)} embeddings in Qdrant"
            }
        else:
            return {
                "status": "error",
                "error": "Failed to store embeddings in vector database"
            }
    except Exception as e:
        logger.error(f"Error storing embeddings: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

