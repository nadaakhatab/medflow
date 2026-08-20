"""
MedFlow PDF Processor Module
Handles PDF validation, magic byte checking, SHA-256 hashing, text extraction,
text cleaning, and section/page-aware chunking for medical RAG indexing.
"""

import io
import re
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import pypdf

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB max limit

def validate_pdf_file(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validates PDF file integrity, mime magic header, extension, size, and encryption state.
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "Selected file is empty."
        
    if len(file_bytes) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE // (1024 * 1024)
        return False, f"File size exceeds maximum upload limit of {max_mb} MB."
        
    if not filename.lower().endswith('.pdf'):
        return False, "Only PDF files are supported."
        
    # Check PDF magic bytes (%PDF-)
    if not file_bytes.startswith(b'%PDF-'):
        return False, "Invalid file format. File is not a valid PDF document."
        
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            # Try empty password
            try:
                decrypted = reader.decrypt('')
                if not decrypted:
                    return False, "PDF document is password protected and cannot be processed."
            except Exception:
                return False, "PDF document is password protected."
                
        if len(reader.pages) == 0:
            return False, "PDF document contains 0 pages."
            
    except Exception as e:
        return False, f"Unable to read this PDF document: {str(e)}"
        
    return True, "Valid PDF"


def calculate_sha256(file_bytes: bytes) -> str:
    """Calculates SHA-256 hash of PDF content for duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()


def clean_extracted_text(text: str) -> str:
    """
    Cleans PDF text by normalizing line breaks and fixing hyphenated line breaks
    without altering clinical semantics.
    """
    if not text:
        return ""
        
    # Replace null bytes and non-printable control chars
    text = text.replace('\x00', ' ')
    
    # Fix hyphenated word breaks at end of line: "hyphen-\nnation" -> "hyphennation"
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Normalize multiple newlines to double newlines (paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize spaces within lines
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
    
    return '\n'.join(lines).strip()


def extract_pages_and_chunks(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
    target_chunk_size: int = 600,
    overlap: int = 100
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page and generates chunks preserving exact page numbers
    and metadata for ChromaDB & BM25 indexing.
    """
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    total_pages = len(reader.pages)
    
    raw_pages_text: List[Tuple[int, str]] = []
    total_extracted_length = 0
    
    for page_idx in range(total_pages):
        page_num = page_idx + 1
        try:
            page_text = reader.pages[page_idx].extract_text() or ""
        except Exception:
            page_text = ""
            
        cleaned_page = clean_extracted_text(page_text)
        if cleaned_page:
            raw_pages_text.append((page_num, cleaned_page))
            total_extracted_length += len(cleaned_page)

    # Check for scanned or image-only PDF
    if total_extracted_length < 50:
        raise ValueError("This PDF contains little or no extractable text. OCR may be required.")

    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1
    upload_time = datetime.now(timezone.utc).isoformat()

    for page_num, page_text in raw_pages_text:
        # Split page text into overlapping character windows or paragraphs
        paragraphs = page_text.split('\n\n')
        current_chunk_str = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(current_chunk_str) + len(para) <= target_chunk_size:
                if current_chunk_str:
                    current_chunk_str += "\n\n" + para
                else:
                    current_chunk_str = para
            else:
                if current_chunk_str:
                    chunk_id = f"UP_{doc_id[:8]}_P{page_num}_C{chunk_counter}"
                    # Infer section title from first line of chunk
                    first_line = current_chunk_str.split('\n')[0][:80]
                    section_heading = f"Page {page_num}: {first_line}" if len(first_line) > 5 else f"Page {page_num} Excerpt"
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "document_name": filename,
                        "page_number": page_num,
                        "section": section_heading,
                        "text_content": current_chunk_str,
                        "disease_category": "Imported Medical PDF",
                        "source_type": "uploaded",
                        "keywords": list(set(re.findall(r'\b[a-zA-Z]{4,}\b', current_chunk_str.lower())))[:10],
                        "upload_timestamp": upload_time
                    })
                    chunk_counter += 1
                    
                    # Apply overlap window
                    overlap_text = current_chunk_str[-overlap:] if len(current_chunk_str) > overlap else ""
                    current_chunk_str = (overlap_text + "\n\n" + para).strip()
                else:
                    current_chunk_str = para

        if current_chunk_str:
            chunk_id = f"UP_{doc_id[:8]}_P{page_num}_C{chunk_counter}"
            first_line = current_chunk_str.split('\n')[0][:80]
            section_heading = f"Page {page_num}: {first_line}" if len(first_line) > 5 else f"Page {page_num} Excerpt"
            
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "document_name": filename,
                "page_number": page_num,
                "section": section_heading,
                "text_content": current_chunk_str,
                "disease_category": "Imported Medical PDF",
                "source_type": "uploaded",
                "keywords": list(set(re.findall(r'\b[a-zA-Z]{4,}\b', current_chunk_str.lower())))[:10],
                "upload_timestamp": upload_time
            })
            chunk_counter += 1

    doc_summary = {
        "document_id": doc_id,
        "filename": filename,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "upload_timestamp": upload_time,
        "status": "Indexed"
    }

    return chunks, doc_summary
