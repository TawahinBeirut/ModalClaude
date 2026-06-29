import modal
from fastapi import Request

ollama_cache = modal.Volume.from_name("ollama-models", create_if_missing=True)
kb_volume    = modal.Volume.from_name("knowledge-base", create_if_missing=True)

KB_PATH = "/data/knowledge_base.json"

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "bash", "ca-certificates")
    .run_commands(
        "apt-get install zstd",
        "curl -fsSL https://ollama.com/install.sh -o /install.sh",
        "chmod +x /install.sh",
        "OLLAMA_NO_PRUNE=1 sh /install.sh",
        "which ollama",
    )
    .pip_install("fastapi[standard]", "httpx", "numpy")
)

app = modal.App("indexer-docs-ollama")

OLLAMA_URL  = "http://localhost:11434"
LLM_MODEL   = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


@app.cls(
    image=image,
    volumes={
        "/root/.ollama": ollama_cache,
        "/data": kb_volume,
    },
    timeout=300,
    scaledown_window=300,
)
class IndexService:

    @modal.enter()
    def start(self):
        import subprocess, time
        subprocess.Popen(["ollama", "serve"])
        time.sleep(2)
        for model in [LLM_MODEL, EMBED_MODEL]:
            subprocess.run(["ollama", "pull", model], check=True)

    @modal.fastapi_endpoint(method="POST")
    async def index_page(self, request: Request):
        import httpx, json

        body    = await request.json()
        source  = body.get("source", "")
        content = body.get("content", "")
        if not content:
            return {"error": "contenu manquant"}

        # ── Extraction métadonnées ────────────────────────────
        prompt = f"""Tu es un extracteur de métadonnées.
Réponds UNIQUEMENT en JSON valide, sans markdown, sans explication.
Format exact :
{{
  "description": "phrase courte décrivant le sujet (pour recherche sémantique)",
  "summary": "résumé en 3-4 phrases",
  "liens_internes": ["url1", "url2"]
}}

Source: {source}
Contenu: {content[:4000]}"""

        async with httpx.AsyncClient(timeout=120) as client:
            meta_res = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            )
            meta_data = meta_res.json()

        try:
            meta = json.loads(meta_data["response"])
        except Exception:
            meta = {"description": content[:200], "summary": "", "liens_internes": []}

        # ── Embedding ─────────────────────────────────────────
        embed_text = f"{meta.get('description', '')} {meta.get('summary', '')} {content[:1000]}"

        async with httpx.AsyncClient(timeout=60) as client:
            embed_res = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": embed_text},
            )
            embed_data = embed_res.json()

        result = {
            "source":         source,
            "content":        content[:3000],
            "description":    meta.get("description", ""),
            "summary":        meta.get("summary", ""),
            "liens_internes": meta.get("liens_internes", []),
            "embeddings":     embed_data.get("embedding", []),
        }

        # ── Persistance dans le volume partagé ───────────────
        try:
            with open(KB_PATH) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []

        existing = [r for r in existing if r.get("source") != source]
        existing.append(result)

        with open(KB_PATH, "w") as f:
            json.dump(existing, f)

        await kb_volume.commit.aio()

        return {
            "source":      source,
            "description": meta.get("description", ""),
            "summary":     meta.get("summary", ""),
        }