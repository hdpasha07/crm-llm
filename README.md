# Local AI CRM Extraction Engine

A local retrieval-augmented CRM assistant. It embeds CRM policy documents with `BAAI/bge-small-en-v1.5`, stores and searches chunks in ChromaDB, and asks a local Ollama model (`qwen-crm`) to return structured JSON.

## Architecture

- `document_loader.py` reads Markdown, text, and JSON files from `data/crm_documents/`, chunks them, and writes embeddings to `crm_chroma_db/`.
- `rag_pipeline.py` is a small command-line example of retrieval and Ollama generation.
- `api.py` exposes `POST /extract` through FastAPI. It retrieves relevant chunks and calls Ollama at `http://localhost:11434`.
- `app.py` provides a Streamlit dashboard with the same retrieval and generation flow.
- The embedding model is downloaded by Hugging Face on first use. Ollama and the `qwen-crm` model are separate local prerequisites.
- `src/train_qwen_crm.py` fine-tunes the pretrained Qwen 2.5 3B Instruct base model on `data/langgraph_agents_train.jsonl` using Unsloth QLoRA. This is fine-tuning, not training a foundation model from random weights.

## Setup

1. Install Python 3.10 or newer and [Ollama](https://ollama.com/). To run inference, Ollama must have a model named `qwen-crm`; follow the training/import section below to recreate this project's fine-tuned model.
2. Create and activate a virtual environment from the project directory:

   ```powershell
   python -m venv final
   .\final\Scripts\Activate.ps1
   ```

   On macOS or Linux, activate it with `source final/bin/activate`.
3. Install Python dependencies:

   ```shell
   python -m pip install -r requirements.txt
   ```
4. The repository includes dummy CRM documents in `data/crm_documents/` and a small prebuilt `crm_chroma_db/` so the demo can run after setup. Replace these with your own sample data as needed.
5. To rebuild the vector database after changing the documents, run:

   ```shell
   python document_loader.py
   ```

   The first run downloads the embedding model. Keep Ollama running for extraction requests. `crm_chroma_db/` is the database used by the app; the separate legacy `vector_db/` directory is ignored.

## Recreate the fine-tuned Ollama model

The trained weights are intentionally not stored in Git. To recreate the model, use a CUDA-capable NVIDIA GPU with enough free VRAM (the training script is tuned for about 6 GB), install the training dependencies, and train from the included JSONL dataset:

```shell
python -m pip install -r requirements-training.txt
python src/train_qwen_crm.py
```

Training downloads `unsloth/Qwen2.5-3B-Instruct` and writes a Q4_K_M GGUF under `model_export/qwen_crm_agent/`. Once training succeeds, create an Ollama model named `qwen-crm` from the generated GGUF. In PowerShell:

```powershell
$gguf = Get-ChildItem .\model_export\qwen_crm_agent -Filter *.gguf -Recurse | Select-Object -First 1
if (-not $gguf) { throw 'No GGUF export found. Check the training output.' }
Set-Content .\Modelfile "FROM $($gguf.FullName)"
ollama create qwen-crm -f .\Modelfile
```

Start Ollama, then run the API or dashboard as described below. The `model_export/` directory and GGUF files are ignored by Git because each person recreates the weights locally.

## Run

Start the FastAPI server from the project directory:

```shell
uvicorn api:app --reload
```

The API is available at `http://127.0.0.1:8000`; interactive docs are at `http://127.0.0.1:8000/docs`. Send a JSON body such as `{"query":"What is the refund policy?"}` to `POST /extract`.

Start the Streamlit dashboard in a separate terminal:

```shell
streamlit run app.py
```

The dashboard opens in your browser. Both interfaces expect the Ollama service at `http://localhost:11434` and the local model name `qwen-crm`.
