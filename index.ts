// index.ts
// Usage : bun index.ts https://url.com ./fichier.html ...

import * as fs from "fs";

const MODAL_INDEXER_URL = process.env.MODAL_INDEXER_URL!;

interface Resource {
  source:         string;
  description:    string;
  summary:        string;
  liens_internes: string[];
  embeddings:     number[];
}

// ── Fetch + nettoyage HTML (100% local) ─────────────────────
async function fetchContent(source: string): Promise<string> {
  let html: string;
  if (source.startsWith("http")) {
    html = await (await fetch(source)).text();
  } else {
    html = fs.readFileSync(source, "utf-8");
  }
  return html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 6000);
}

// ── Appel Modal (Ollama dans le container) ───────────────────
async function indexViaModal(source: string, content: string): Promise<Resource> {
  const res = await fetch(MODAL_INDEXER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, content }),
  });
  if (!res.ok) throw new Error(`Modal a répondu ${res.status}`);
  return res.json() as Promise<Resource>;
}

// ── DB locale ────────────────────────────────────────────────
function loadDB(): Resource[] {
  return fs.existsSync("knowledge_base.json")
    ? JSON.parse(fs.readFileSync("knowledge_base.json", "utf-8"))
    : [];
}

function saveDB(resources: Resource[]) {
  fs.writeFileSync("knowledge_base.json", JSON.stringify(resources, null, 2));
  console.log(`   💾 knowledge_base.json (${resources.length} ressources)`);
}

// ── Main ─────────────────────────────────────────────────────
const sources = process.argv.slice(2);

if (!MODAL_INDEXER_URL) {
  console.error("❌ Variable MODAL_INDEXER_URL manquante dans .env");
  process.exit(1);
}
if (!sources.length) {
  console.error("Usage: bun index.ts <url|fichier.html> ...");
  process.exit(1);
}

const db       = loadDB();
const existing = new Set(db.map(r => r.source));

for (const source of sources) {
  if (existing.has(source)) {
    console.log(`⏭  Déjà indexé : ${source}`);
    continue;
  }
  console.log(`\n📄 ${source}`);
  try {
    const content  = await fetchContent(source);
    console.log(`   ✓ Contenu (${content.length} chars)`);

    const resource = await indexViaModal(source, content);
    console.log(`   ✓ "${resource.description.slice(0, 60)}..."`);
    console.log(`   ✓ Embedding (${resource.embeddings.length} dims)`);

    db.push(resource);
    saveDB(db);
  } catch (err) {
    console.error(`   ❌ ${err}`);
  }
}

console.log(`\n🎉 Terminé — ${db.length} ressource(s) dans knowledge_base.json`);