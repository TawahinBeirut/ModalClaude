# modal_app.py
import modal
from fastapi import Request
from fastapi.responses import JSONResponse

ollama_cache = modal.Volume.from_name("ollama-models", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .add_local_file("knowledge_base.json", "/root/knowledge_base.json",copy=True) 
    .apt_install("curl", "bash", "ca-certificates")
    .run_commands(
        "apt-get install zstd",
        "curl -fsSL https://ollama.com/install.sh -o /install.sh",
        "chmod +x /install.sh",
        "OLLAMA_NO_PRUNE=1 sh /install.sh",
        "which ollama",  # vérifie que l'install a marché
    )
    .pip_install("fastapi[standard]", "httpx", "numpy")
)
app = modal.App("chatbot-docs-ollama")

OLLAMA_URL  = "http://localhost:11434"
LLM_MODEL   = "llama3.2"
EMBED_MODEL = "nomic-embed-text"


def start_ollama_and_pull():
    import subprocess, time
    subprocess.Popen(["ollama", "serve"])
    time.sleep(2)
    for model in [LLM_MODEL, EMBED_MODEL]:
        subprocess.run(["ollama", "pull", model], check=True)


@app.function(
    image=image,
    volumes={"/root/.ollama": ollama_cache},
    # gpu="any",  # optionnel
    timeout=300,
    scaledown_window=300
)
@modal.fastapi_endpoint(method="POST")
async def chat(request: Request):
    import httpx, json, numpy as np

    start_ollama_and_pull()

    body     = await request.json()
    question = body.get("question", "")
    if not question:
        return JSONResponse({"error": "champ 'question' manquant"}, status_code=400)

    with open("/root/knowledge_base.json") as f:
        resources = json.load(f)

    if not resources:
        return JSONResponse({"error": "base vide — lance d'abord bun index.ts"}, status_code=400)

    # ── Embed la question ────────────────────────────────────
    async with httpx.AsyncClient(timeout=60) as client:
        embed_res = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": question},
        )
        q_vec = np.array(embed_res.json().get("embedding", []))

    # ── Similarité cosinus ───────────────────────────────────
    def cosine(a, b):
        b = np.array(b)
        # aligne les dimensions si différentes (sécurité)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    scored = sorted(
        [(cosine(q_vec, r["embeddings"]), r) for r in resources],
        reverse=True,
    )[:3]

    # ── Réponse LLM ──────────────────────────────────────────
    context = "\n\n".join(
        f"[score {s:.2f}] {r['description']}\n"
        f"Résumé : {r['summary']}\n"
        f"Source : {r['source']}"
        for s, r in scored
    )

    prompt = f"""Tu es un assistant spécialisé en logistique et Python.
Explique en 2-3 phrases pourquoi les ressources ci-dessous répondent à la question,
puis liste les sources. Si elles ne correspondent pas bien, dis-le franchement.

Question : {question}

Ressources :
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
        "sources":  [r["source"] for _, r in scored],
    }