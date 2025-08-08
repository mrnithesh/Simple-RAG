# PDF Chat Assistant 📄

A simplified RAG (Retrieval-Augmented Generation) chatbot that uses local LLMs via Ollama to chat with a single hardcoded PDF document. This is a streamlined version focused on simplicity and local processing.

## Features

- 🏠 **Local Processing**: Uses Ollama for local LLM inference
- 📄 **Hardcoded Document**: Works with a single PDF file (no upload needed)
- 🔍 **Vector Search**: ChromaDB with local embeddings
- 💬 **Simple Chat Interface**: Clean Streamlit UI
- 🔄 **Auto-Processing**: Automatically loads and processes your PDF on startup

## Prerequisites

1. Install Python 3.10+
2. Install Ollama: download from [ollama.com](https://ollama.com)
3. Pull a model (default is Gemma):
   - `ollama pull gemma3:1b`
   - To use Meta Llama instead: `ollama pull llama3.2` (or `llama3.1`)
4. Add your PDF at `documents/document.pdf`

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

1. Place your PDF at `documents/document.pdf`
2. Start the app: `streamlit run main.py`
3. In the sidebar, verify "Chunks indexed" is > 0 (otherwise click "Reload Document")
4. Start chatting

The system will automatically:
- Load your PDF on startup
- Process it into searchable chunks
- Create vector embeddings
- Enable chat functionality



Make sure to pull the model first: `ollama pull <model-name>`

### Change the local model (Gemma ↔ Llama)

1. Pull the desired model with Ollama:
   Examples:
   - Gemma: `ollama pull gemma3:1b`
   - Llama: `ollama pull llama3.2`
   - Mistral: `ollama pull mistral`
   - Codellama: `ollama pull codellama`
   - Phi3: `ollama pull phi3`

2. Edit the model name in code. Open `src/ui/app.py` and change:

   ```python
   # src/ui/app.py
   model_name = "gemma3:1b"  # change to "llama3.2" or another pulled model
   ```

3. Restart the app: `streamlit run main.py`

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