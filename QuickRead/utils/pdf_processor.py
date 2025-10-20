
# utils/pdf_processor.py
import PyPDF2
import pdfplumber
from typing import Tuple, List
import io

class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_text_with_pypdf2(self, pdf_file) -> Tuple[str, int]:
        """Extract text using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text, len(pdf_reader.pages)
        except Exception as e:
            raise Exception(f"PyPDF2 extraction failed: {str(e)}")
    
    def extract_text_with_pdfplumber(self, pdf_file) -> Tuple[str, int]:
        """Extract text using pdfplumber (more accurate for some PDFs)"""
        try:
            text = ""
            page_count = 0
            with pdfplumber.open(pdf_file) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text, page_count
        except Exception as e:
            raise Exception(f"PDFPlumber extraction failed: {str(e)}")
    
    def extract_text(self, pdf_file) -> Tuple[str, int, dict]:
        """Main method to extract text from PDF with best available method"""
        # Reset file pointer
        pdf_file.seek(0)
        
        # Try pdfplumber first (more accurate)
        try:
            text, page_count = self.extract_text_with_pdfplumber(pdf_file)
            if text.strip():
                return text, page_count, {"method": "pdfplumber", "status": "success"}
        except Exception as e:
            print(f"PDFPlumber failed: {e}")
        
        # Fallback to PyPDF2
        pdf_file.seek(0)
        try:
            text, page_count = self.extract_text_with_pypdf2(pdf_file)
            if text.strip():
                return text, page_count, {"method": "pypdf2", "status": "success"}
        except Exception as e:
            print(f"PyPDF2 failed: {e}")
        
        # Both methods failed
        raise Exception("Could not extract text from PDF. The file might be scanned or corrupted.")

# Test function
def test_pdf_processor():
    processor = PDFProcessor()
    print("PDF Processor initialized successfully!")