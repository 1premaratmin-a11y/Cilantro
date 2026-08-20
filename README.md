# Cilantro Recipe Assistant

A Streamlit chatbot that searches a ChromaDB collection of 200 recipes. It uses Ollama embeddings, OpenAI for answers, cites its source, and refuses unrelated questions.

Chat settings include RAG ON/OFF, source visibility, and microphone input.

## Versions

- `cilantro_app_v1.py`: Ollama answers, old prompt, no voice input
- `cilantro_app_v2.py`: Ollama answers, old prompt, voice input
- `cilantro_app_v3.py`: OpenAI answers, tighter prompt, voice input
- `cilantro_app.py`: current app, matching version 3

## Project files

- `.streamlit/config.toml`: Streamlit colors and static-file settings
- `assets/`: Cilantro logo
- `final_project_db/`: Saved Chroma recipe database
- `static/`: Browser-accessible source recipe PDF
- `.gitignore`: Keeps local and secret files out of Git
- `cilantro_pipeline.py`: Loads, embeds, and stores the recipes
- `embedding_helper.py`: Creates embeddings with Ollama
- `requirements.txt`: Python packages required by the project

## Run

Install [Ollama](https://ollama.com/) and Python 3.11 or newer. Add your key to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-key-here"
```

Then run:

```powershell
ollama pull nomic-embed-text
pip install -r requirements.txt
streamlit run cilantro_app.py
```

The recipe database and source PDF are included. To rebuild the database, run:

```powershell
python cilantro_pipeline.py
```
