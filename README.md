# PDF Chat Assistant 📄

A simplified RAG (Retrieval-Augmented Generation) chatbot that uses local LLMs via Ollama to chat with a single hardcoded PDF document. This is a streamlined version focused on simplicity and local processing.

## Features

- 🏠 **Local Processing**: Uses Ollama for local LLM inference
- 📄 **Hardcoded Document**: Works with a single PDF file (no upload needed)
- 🔍 **Vector Search**: ChromaDB with local embeddings
- 💬 **Simple Chat Interface**: Clean Streamlit UI
- 🔄 **Auto-Processing**: Automatically loads and processes your PDF on startup

## Prerequisites

1. **Install Ollama**: Download and install from [ollama.com](https://ollama.com)
2. **Pull a Model**: Run `ollama pull gemma3:1b` (or any other supported model)
3. **Add Your PDF**: Place your PDF file at `documents/document.pdf`

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Your Document**:
   ```bash
   # Place your PDF file in the documents folder
   cp /path/to/your/document.pdf documents/document.pdf
   ```

3. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Place your PDF**: Put your PDF file at `documents/document.pdf`
2. **Start the app**: Run `streamlit run main.py`
3. **Start Chatting**: Ask questions about your document!

The system will automatically:
- Load your PDF on startup
- Process it into searchable chunks
- Create vector embeddings
- Enable chat functionality



Make sure to pull the model first: `ollama pull <model-name>`

## Project Structure

```
RAG/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── documents/
│   └── document.pdf       # Your PDF file goes here
├── src/
│   ├── models/
│   │   └── agent.py       # Simple Ollama-based agent
│   ├── ui/
│   │   └── app.py         # Streamlit interface
│   ├── utils/
│   │   └── document_utils.py  # Document processing
│   └── config/
│       └── constants.py   # Configuration
└── chroma_db/             # Vector database storage
```


## Changing Documents

To use a different PDF:
1. Replace `documents/document.pdf` with your new file
2. Click "🔄 Reload Document" in the sidebar, or
3. Restart the application

## Troubleshooting

- **"PDF file not found"**: Make sure your PDF is at `documents/document.pdf`
- **Model not found**: Make sure you've pulled the model with `ollama pull <model-name>`
- **Ollama not running**: Start Ollama service
- **Empty responses**: Check if the PDF was processed correctly (look for success message on startup) 