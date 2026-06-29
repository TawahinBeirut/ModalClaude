import modal
from fastapi import Request
from fastapi.responses import JSONResponse

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

app = modal.App("chatbot-docs-ollama")

OLLAMA_URL  = "http://localhost:11434"
LLM_MODEL   = "mistral"
EMBED_MODEL = "mxbai-embed-large"


def wait_for_ollama():
    """Attend qu'ollama soit vraiment prêt avant de pull."""
    import httpx, time
    for _ in range(30):
        try:
            httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return  # ollama répond → ok
        except Exception:
            time.sleep(2)
    raise RuntimeError("Ollama n'a pas démarré après 60s")


@app.cls(
    image=image,
    volumes={
        "/root/.ollama": ollama_cache,
        "/data": kb_volume,
    },
    timeout=300,
    scaledown_window=300,
)
class ChatService:

    @modal.enter()
    def start(self):
        import subprocess
        subprocess.Popen(["ollama", "serve"])
        wait_for_ollama()
        for model in [LLM_MODEL, EMBED_MODEL]:
            subprocess.run(["ollama", "pull", model], check=True)

    @modal.fastapi_endpoint(method="POST")
    async def chat(self, request: Request):
        import httpx, json, numpy as np

        body     = await request.json()
        question = body.get("question", "")
        if not question:
            return JSONResponse({"error": "champ 'question' manquant"}, status_code=400)

        # ── Lecture depuis le volume partagé ────────────────
        await kb_volume.reload.aio()
        try:
            with open(KB_PATH) as f:
                resources = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            resources = []

        if not resources:
            return JSONResponse({"error": "base vide — lance d'abord l'indexer"}, status_code=400)

        # ── Embed la question ────────────────────────────────
        async with httpx.AsyncClient(timeout=60) as client:
            embed_res = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": question},
            )
            q_vec = np.array(embed_res.json().get("embedding", []))

        # ── Similarité cosinus ───────────────────────────────
        def cosine(a, b):
            b = np.array(b)
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            d = np.linalg.norm(a) * np.linalg.norm(b)
            return float(np.dot(a, b) / d) if d > 0 else 0.0

        scored = sorted(
            [(cosine(q_vec, r["embeddings"]), r) for r in resources],
            reverse=True,
        )[:5]

        MIN_SCORE = 0.5
        relevant  = [(s, r) for s, r in scored if s >= MIN_SCORE]
        if not relevant:
            relevant = scored[:2]

        # ── Contexte avec contenu brut ───────────────────────
        context = "\n\n---\n\n".join(
            f"Source : {r['source']}\n\n"
            f"{r.get('content', r.get('description', ''))}"
            for _, r in relevant
        )

        prompt = f"""Tu es un assistant de documentation technique.
Réponds en 2-3 phrases claires et directes, basées uniquement sur les sources fournies.
Cite le lien exact de la source utilisée à la fin de ta réponse, tel qu'il apparaît dans "Source :".
Ne génère pas de lien de toi-même.

Question : {question}

Sources :
{context}"""

        async with httpx.AsyncClient(timeout=120) as client:
            llm_res = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            )
            answer = llm_res.json().get("response", "")

        return {
            "question": question,
            "answer":   answer,
            "sources":  [r["source"] for _, r in relevant],
            "scores":   [round(s, 3) for s, _ in relevant],
        }