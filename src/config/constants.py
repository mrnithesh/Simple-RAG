import os

CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Hardcoded document settings
FIXED_PROJECT_NAME = "default_documents"
DOCUMENTS_DIR = "documents"
FIXED_PDF_PATH = os.path.join(DOCUMENTS_DIR, "document.pdf")

# Create documents directory
os.makedirs(DOCUMENTS_DIR, exist_ok=True) 