# ManasAI

ManasAI is a digital sage and question-answering system built on the Gaudiya Vaishnava tradition, powered by LlamaIndex, Ollama, and modern LLMs. It provides deep, context-aware answers to philosophical and devotional questions, drawing from a curated corpus of sacred texts and user-provided documents.

## Features
- **Contextual QA Engine:** Uses LlamaIndex to index and search documents in the `data/` folder, updating the index automatically when new files are added.
- **Persona System:** Answers are generated in the voice of a mystical sage, with strict rules for quoting scripture and providing purports.
- **Hybrid Prompting:** Combines context from indexed documents with the model's own knowledge, using a custom prompt template.
- **Flask API Server:** Exposes a `/api/chat` endpoint for integration with web or other client applications.
- **Ollama Integration:** Uses local Ollama models (e.g., gemma3:4b) for both LLM and embedding tasks.
- **Automatic Index Persistence:** Index is saved to the `storage/` directory and loaded on startup.
- **Requirements and Environment:** All dependencies are tracked in `requirements.txt`. The `venv/` and `data/` folders are excluded from version control.

## Usage

### 1. Setup
- Clone the repository.
- Create and activate a Python virtual environment:
  ```bash
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On Mac/Linux:
  source venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Ensure Ollama is installed and running, with the required models downloaded (e.g., `gemma3:4b`).

### 2. Add Documents
- Place text or PDF files in the `data/` folder. The system will automatically index new files on startup.

### 3. Run the App
- To run the CLI/terminal app:
  ```bash
  python app.py
  ```
- To run the API server:
  ```bash
  python api_server.py
  ```
- The API server exposes `/api/chat` for POST requests with a JSON body: `{ "prompt": "your question" }`

## File Structure
```
ManasAI/
├── app.py            # Main CLI app for QA and indexing
├── api_server.py     # Flask API server for chat
├── data/             # Folder for user documents (excluded from git)
├── storage/          # Persistent index storage (excluded from git)
├── requirements.txt  # Python dependencies
├── .gitignore        # Excludes venv/, data/, storage/, etc.
└── README.md         # This file
```

## Persona & Prompting
- The system prompt and QA template enforce a devotional, philosophical tone and strict quoting rules.
- Answers are always comprehensive, with scriptural references and purports when possible.

## Contributing
- Fork and clone the repo.
- Submit pull requests for improvements, bug fixes, or new features.

## License
MIT License

## Maintainers
Mantra AI Solutions