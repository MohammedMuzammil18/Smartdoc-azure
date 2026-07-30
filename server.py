"""
SmartDoc Search — Flask Backend
================================
Primary:  Azure AI Search (when AZURE_SEARCH_ENDPOINT is configured)
Fallback: Local Sentence-Transformers semantic search
Also:     ML document classifier, user auth (SQLite), document CRUD
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np
from dotenv import load_dotenv
import sqlite3
import hashlib
import logging

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("smartdoc")

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

AZURE_SEARCH_ENDPOINT   = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
AZURE_SEARCH_API_KEY    = os.getenv("AZURE_SEARCH_API_KEY", "").strip()
AZURE_SEARCH_INDEX      = os.getenv("AZURE_SEARCH_INDEX", "documents").strip()
AZURE_SEARCH_ADMIN_KEY  = os.getenv("AZURE_SEARCH_ADMIN_KEY", AZURE_SEARCH_API_KEY).strip()
AZURE_SEARCH_SEMANTIC   = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "").strip()
AZURE_SEARCH_API_VER    = os.getenv("AZURE_SEARCH_API_VERSION", "2023-11-01").strip()

AZURE_OPENAI_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "https://haseebuddin.openai.azure.com/").strip()
AZURE_OPENAI_KEY        = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o").strip()

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)

# ── Database Initialization ────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Search history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── ML Classifier ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "smart_doc_classifier.pkl")

logger.info("Loading ML model from: %s", MODEL_PATH)
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("[OK] ML model loaded successfully!")
except Exception as e:
    logger.warning("[WARN] Error loading ML model: %s", e)
    model = None

# ── Azure AI Search Client ─────────────────────────────────────────────────────
azure_configured = False
azure_client = None

if AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY:
    try:
        from azure.search.documents import SearchClient
        from azure.search.documents.models import QueryType
        from azure.core.credentials import AzureKeyCredential
        azure_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
        )
        azure_configured = True
        logger.info("[OK] Azure AI Search client initialized. Index: %s", AZURE_SEARCH_INDEX)
    except ImportError:
        logger.warning("[WARN] azure-search-documents not installed. Run: pip install azure-search-documents")
    except Exception as e:
        logger.warning("[WARN] Azure AI Search init error: %s", e)
else:
    logger.info("[INFO] Azure AI Search not configured — using local semantic search.")

# ── Local Semantic Search ──────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    _semantic_libs_available = True
except ImportError:
    logger.warning("[WARN] sentence_transformers or sklearn not found. Semantic search unavailable.")
    _semantic_libs_available = False

LOCAL_INDEX_PATH = os.path.join(BASE_DIR, "local_search_index.pkl")
search_model     = None
local_documents  = []
local_embeddings = None
semantic_configured = False

logger.info("Loading local semantic index from: %s", LOCAL_INDEX_PATH)
try:
    if _semantic_libs_available and os.path.exists(LOCAL_INDEX_PATH):
        with open(LOCAL_INDEX_PATH, "rb") as f:
            index_data = pickle.load(f)
        local_documents  = index_data.get("documents", [])
        local_embeddings = index_data.get("embeddings")
        model_name       = index_data.get("model_name", "all-MiniLM-L6-v2")
        logger.info("Loading semantic model: %s …", model_name)
        search_model     = SentenceTransformer(model_name)
        semantic_configured = True
        logger.info("[OK] Local semantic search loaded with %d documents.", len(local_documents))
    else:
        logger.warning("[WARN] local_search_index.pkl not found. Run build_local_index.py first.")
except Exception as e:
    logger.warning("[WARN] Error loading semantic index: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Health check — reports status of all subsystems."""
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "semantic_search_ready": semantic_configured,
        "azure_search_configured": azure_configured,
        "document_count": len(local_documents),
        "azure_index": AZURE_SEARCH_INDEX if azure_configured else None
    })


# ── Auth ───────────────────────────────────────────────────────────────────────

@app.route("/register", methods=["POST"])
def register():
    """Register a new user."""
    data     = request.get_json(force=True) or {}
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash)
        )
        conn.commit()
        conn.close()
        logger.info("New user registered: %s", email)
        return jsonify({"success": True, "message": "User registered successfully!"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already exists."}), 409
    except Exception as e:
        logger.error("Register error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    """Authenticate a user."""
    data     = request.get_json(force=True) or {}
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "SELECT id, name, email, created_at FROM users WHERE email = ? AND password_hash = ?",
            (email, password_hash)
        )
        user = c.fetchone()
        conn.close()
        if user:
            return jsonify({
                "success": True,
                "user": {"id": user[0], "name": user[1], "email": user[2], "created_at": user[3]}
            })
        return jsonify({"error": "Invalid email or password."}), 401
    except Exception as e:
        logger.error("Login error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.route("/admin/users", methods=["GET"])
def get_users():
    """Return all registered users (admin only)."""
    try:
        conn  = sqlite3.connect(DB_PATH)
        c     = conn.cursor()
        c.execute("SELECT id, name, email, created_at FROM users ORDER BY created_at DESC")
        users = [{"id": r[0], "name": r[1], "email": r[2], "created_at": r[3]} for r in c.fetchall()]
        conn.close()
        return jsonify({"success": True, "users": users})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/documents", methods=["GET"])
def get_documents():
    """Return all documents in the local semantic search index."""
    try:
        docs = [
            {
                "id":       doc.get("id", ""),
                "title":    doc.get("title", doc.get("id", "")),
                "text":     doc.get("text", ""),
                "category": doc.get("category", "General"),
                "snippet":  doc.get("snippet", doc.get("text", "")[:200])
            }
            for doc in local_documents
        ]
        return jsonify({"success": True, "documents": docs, "count": len(docs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Azure Status ───────────────────────────────────────────────────────────────

@app.route("/azure/status", methods=["GET"])
def azure_status():
    """Return Azure AI Search service configuration and index statistics."""
    if not azure_configured:
        return jsonify({
            "configured": False,
            "message": "Azure AI Search is not configured. Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY in .env"
        })
    try:
        import requests as req_lib
        stats_url = (
            f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/stats"
            f"?api-version={AZURE_SEARCH_API_VER}"
        )
        headers = {"api-key": AZURE_SEARCH_API_KEY, "Content-Type": "application/json"}
        resp    = req_lib.get(stats_url, headers=headers, timeout=10)
        if resp.ok:
            stats = resp.json()
            return jsonify({
                "configured": True,
                "endpoint":   AZURE_SEARCH_ENDPOINT,
                "index":      AZURE_SEARCH_INDEX,
                "api_version": AZURE_SEARCH_API_VER,
                "document_count": stats.get("documentCount", "N/A"),
                "storage_bytes":  stats.get("storageSize", "N/A")
            })
        return jsonify({
            "configured": True,
            "endpoint":   AZURE_SEARCH_ENDPOINT,
            "index":      AZURE_SEARCH_INDEX,
            "message":    f"Index stats request returned {resp.status_code}"
        })
    except Exception as e:
        return jsonify({"configured": True, "endpoint": AZURE_SEARCH_ENDPOINT, "error": str(e)})


# ── Azure OpenAI / Foundry Integration ─────────────────────────────────────────

@app.route("/ai/ask", methods=["POST"])
def ai_ask():
    """Query Azure OpenAI / Foundry model using configured credentials."""
    if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
        return jsonify({"error": "Azure OpenAI / Foundry credentials not configured in .env"}), 500

    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "").strip() or data.get("query", "").strip()
    deployment = data.get("deployment", AZURE_OPENAI_DEPLOYMENT)

    if not prompt:
        return jsonify({"error": "No prompt or query provided."}), 400

    import requests as req_lib
    url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version=2024-02-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    payload = {
        "messages": [
            {"role": "system", "content": "You are SmartDoc AI assistant helping answer questions based on enterprise documents."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    try:
        resp = req_lib.post(url, json=payload, headers=headers, timeout=15)
        if resp.ok:
            result = resp.json()
            answer = result["choices"][0]["message"]["content"]
            return jsonify({"success": True, "answer": answer, "deployment": deployment})
        elif resp.status_code == 404:
            return jsonify({
                "success": False,
                "error": f"Deployment '{deployment}' not found in Azure OpenAI service. Please create a model deployment (e.g. gpt-4o) in your Azure AI Foundry / Azure Portal.",
                "endpoint": AZURE_OPENAI_ENDPOINT
            }), 404
        else:
            return jsonify({"success": False, "error": resp.text, "status_code": resp.status_code}), resp.status_code
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



# ── Azure Bulk Import ──────────────────────────────────────────────────────────

@app.route("/azure/bulk-import", methods=["POST"])
def azure_bulk_import():
    """Upload all local documents to the Azure AI Search index."""
    if not azure_configured:
        return jsonify({"error": "Azure AI Search is not configured."}), 503

    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential

        # Use admin key for write operations
        admin_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        )

        batch = [
            {
                "@search.action": "mergeOrUpload",
                "id":       doc.get("id", ""),
                "content":  doc.get("text", ""),
                "category": doc.get("category", "General"),
                "snippet":  doc.get("snippet", doc.get("text", "")[:200])
            }
            for doc in local_documents
            if doc.get("id")
        ]

        if not batch:
            return jsonify({"success": True, "message": "No documents to upload.", "count": 0})

        # Upload in batches of 1000 (Azure SDK limit)
        total_uploaded = 0
        for i in range(0, len(batch), 1000):
            chunk = batch[i:i + 1000]
            admin_client.upload_documents(documents=chunk)
            total_uploaded += len(chunk)

        logger.info("Bulk import: %d documents uploaded to Azure index '%s'.", total_uploaded, AZURE_SEARCH_INDEX)
        return jsonify({"success": True, "count": total_uploaded, "index": AZURE_SEARCH_INDEX})
    except Exception as e:
        logger.error("Bulk import error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── ML Classifier ──────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """Use ML model to classify a text snippet into HR | IT | Finance."""
    if model is None:
        return jsonify({"error": "ML model not loaded."}), 500

    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"category": None, "message": "No text provided."})

    try:
        prediction = model.predict([text])[0]
        return jsonify({"category": prediction, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


# ── Autocomplete / Suggestions ─────────────────────────────────────────────────

@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    """
    Return autocomplete suggestions for a partial query.
    Tries Azure AI Search suggestions first; falls back to local keyword matching.
    """
    data  = request.get_json(force=True) or {}
    query = data.get("query", "").strip().lower()
    top   = min(int(data.get("top", 5)), 10)

    if not query or len(query) < 2:
        return jsonify({"suggestions": []})

    suggestions = []

    # 1. Try Azure suggestions
    if azure_configured:
        try:
            import requests as req_lib
            suggest_url = (
                f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/suggest"
                f"?api-version={AZURE_SEARCH_API_VER}"
            )
            headers = {"api-key": AZURE_SEARCH_API_KEY, "Content-Type": "application/json"}
            payload = {
                "search": query,
                "suggesterName": "sg",
                "top": top,
                "select": "content,category"
            }
            resp = req_lib.post(suggest_url, json=payload, headers=headers, timeout=5)
            if resp.ok:
                azure_sugg = resp.json().get("value", [])
                suggestions = [s.get("@search.text", "") for s in azure_sugg if s.get("@search.text")]
        except Exception as e:
            logger.debug("Azure suggest error: %s", e)

    # 2. Fallback: match against local document titles / snippets
    if not suggestions:
        seen = set()
        for doc in local_documents:
            text    = doc.get("text", "")
            snippet = doc.get("snippet", "")
            for source in (text, snippet):
                words = source.lower().split()
                for i, word in enumerate(words):
                    if word.startswith(query):
                        phrase = " ".join(words[i:i + 4])
                        if phrase not in seen:
                            seen.add(phrase)
                            suggestions.append(phrase)
                        if len(suggestions) >= top:
                            break
                if len(suggestions) >= top:
                    break
            if len(suggestions) >= top:
                break

    return jsonify({"suggestions": suggestions[:top]})


# ── Search ─────────────────────────────────────────────────────────────────────

@app.route("/search", methods=["POST"])
def search():
    """
    Search documents.
    Priority:
      1. Azure AI Search (if configured)
      2. Local Sentence-Transformers semantic search
      3. 503 → frontend falls back to client-side search
    """
    data    = request.get_json(force=True) or {}
    query   = data.get("query", "").strip()
    filters = data.get("filters", [])   # e.g. ["HR", "IT"]
    top     = min(int(data.get("top", 10)), 50)
    page    = max(int(data.get("page", 1)), 1)
    sort_by = data.get("sort", "relevance")  # "relevance" | "date"

    if not query:
        return jsonify({"results": [], "azure": False, "message": "No query provided."})

    # ── (A) Azure AI Search ────────────────────────────────────────────────────
    if azure_configured and azure_client:
        try:
            from azure.search.documents.models import QueryType
            search_kwargs = {
                "search_text":   query,
                "top":           top,
                "skip":          (page - 1) * top,
                "query_type":    QueryType.SIMPLE,
                "include_total_count": True,
            }

            # Semantic ranking if configured
            if AZURE_SEARCH_SEMANTIC:
                search_kwargs["query_type"]              = QueryType.SEMANTIC
                search_kwargs["semantic_configuration_name"] = AZURE_SEARCH_SEMANTIC
                search_kwargs["query_caption"]           = "extractive"

            # Category filter
            if filters:
                filter_expr = " or ".join([f"category eq '{f}'" for f in filters])
                search_kwargs["filter"] = filter_expr

            # Sort
            if sort_by == "date":
                search_kwargs["order_by"] = ["lastUpdated desc"]

            results_iter = azure_client.search(**search_kwargs)
            hits         = []
            for r in results_iter:
                hits.append({
                    "id":       r.get("id", ""),
                    "text":     r.get("content", r.get("text", "")),
                    "category": r.get("category", ""),
                    "snippet":  r.get("snippet", r.get("content", "")[:200]),
                    "score":    r.get("@search.score", 0.0),
                    "source":   "azure"
                })

            logger.info("Azure search for '%s': %d results.", query, len(hits))
            return jsonify({
                "results": hits,
                "count":   len(hits),
                "azure":   True,
                "source":  "azure"
            })
        except Exception as e:
            logger.warning("Azure search failed, falling back to local: %s", e)

    # ── (B) Local Semantic Search ──────────────────────────────────────────────
    if semantic_configured and search_model and local_embeddings is not None:
        try:
            query_embedding = search_model.encode([query], convert_to_numpy=True)
            similarities    = cosine_similarity(query_embedding, np.asarray(local_embeddings))[0]

            if filters:
                valid_indices = [i for i, doc in enumerate(local_documents) if doc.get("category") in filters]
            else:
                valid_indices = list(range(len(local_documents)))

            if not valid_indices:
                return jsonify({"results": [], "count": 0, "azure": True})

            sorted_indices = sorted(valid_indices, key=lambda i: similarities[i], reverse=True)

            # Sort by date if requested (approximation using doc index order)
            if sort_by == "date":
                sorted_indices = list(reversed(sorted_indices))

            # Pagination
            start      = (page - 1) * top
            top_indices = sorted_indices[start:start + top]

            hits = []
            for idx in top_indices:
                doc   = local_documents[idx]
                score = float(similarities[idx])
                hits.append({
                    "id":       doc.get("id", ""),
                    "title":    doc.get("title", doc.get("id", "")),
                    "text":     doc.get("text", ""),
                    "category": doc.get("category", ""),
                    "snippet":  doc.get("snippet", doc.get("text", "")[:200]),
                    "score":    score,
                    "source":   "local_semantic"
                })

            return jsonify({
                "results":    hits,
                "count":      len(hits),
                "total":      len(sorted_indices),
                "page":       page,
                "azure":      True   # Keeps existing frontend panel active
            })
        except Exception as e:
            logger.error("Local search error: %s", e)
            return jsonify({"error": str(e), "azure": False}), 500

    # ── (C) Not configured ─────────────────────────────────────────────────────
    return jsonify({
        "results": [],
        "azure":   False,
        "message": "Search index not configured. Run build_local_index.py first."
    }), 503


# ── Document CRUD ──────────────────────────────────────────────────────────────

@app.route("/add-doc", methods=["POST"])
def add_doc():
    """Add a new document to the local semantic index (and Azure if configured)."""
    global local_documents, local_embeddings

    if not semantic_configured or search_model is None:
        return jsonify({"error": "Semantic search is not configured."}), 503

    data     = request.get_json(force=True) or {}
    doc_id   = data.get("id")
    text     = data.get("text", "").strip()
    category = data.get("category", "General")
    snippet  = data.get("snippet", text[:200])

    if not text or not doc_id:
        return jsonify({"error": "Text and id are required."}), 400

    try:
        new_embedding = search_model.encode([text])
        new_doc = {"id": doc_id, "text": text, "category": category, "snippet": snippet}
        local_documents.append(new_doc)

        if local_embeddings is None:
            local_embeddings = new_embedding
        else:
            local_embeddings = np.vstack([local_embeddings, new_embedding])

        _save_index()

        # Mirror to Azure if configured
        if azure_configured:
            _azure_upsert_doc(new_doc)

        logger.info("Document %s added.", doc_id)
        return jsonify({"success": True, "message": "Document indexed successfully."})
    except Exception as e:
        logger.error("Add doc error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/update-doc", methods=["POST"])
def update_doc():
    """Update a document in the local semantic index."""
    global local_documents, local_embeddings

    if not semantic_configured or search_model is None:
        return jsonify({"error": "Semantic search is not configured."}), 503

    data     = request.get_json(force=True) or {}
    doc_id   = data.get("id")
    text     = data.get("text", "").strip()
    category = data.get("category", "General")
    snippet  = data.get("snippet", text[:200])

    if not text or not doc_id:
        return jsonify({"error": "Text and id are required."}), 400

    try:
        idx = next((i for i, d in enumerate(local_documents) if d.get("id") == doc_id), None)
        if idx is None:
            return jsonify({"error": "Document not found."}), 404

        assert local_embeddings is not None
        local_documents[idx]      = {"id": doc_id, "text": text, "category": category, "snippet": snippet}
        new_embedding             = search_model.encode([text], convert_to_numpy=True)
        local_embeddings[idx]     = new_embedding[0]

        _save_index()

        if azure_configured:
            _azure_upsert_doc(local_documents[idx])

        logger.info("Document %s updated.", doc_id)
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Update doc error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/delete-doc", methods=["POST"])
def delete_doc():
    """Delete a document from the local semantic index (and Azure if configured)."""
    global local_documents, local_embeddings

    if not semantic_configured or search_model is None:
        return jsonify({"error": "Semantic search is not configured."}), 503

    data   = request.get_json(force=True) or {}
    doc_id = data.get("id")

    if not doc_id:
        return jsonify({"error": "Document id is required."}), 400

    try:
        idx = next((i for i, d in enumerate(local_documents) if d.get("id") == doc_id), None)
        if idx is None:
            return jsonify({"error": "Document not found."}), 404

        del local_documents[idx]
        assert local_embeddings is not None
        local_embeddings = np.delete(np.asarray(local_embeddings), idx, axis=0)

        _save_index()

        if azure_configured:
            _azure_delete_doc(doc_id)

        logger.info("Document %s deleted.", doc_id)
        return jsonify({"success": True})
    except Exception as e:
        logger.error("Delete doc error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Search History ─────────────────────────────────────────────────────────────

@app.route("/search-history", methods=["POST"])
def save_search_history():
    """Save a search query to history for a user."""
    data    = request.get_json(force=True) or {}
    user_id = data.get("user_id")
    query   = data.get("query", "").strip()

    if not user_id or not query:
        return jsonify({"error": "user_id and query are required."}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        # Keep only last 50 per user
        c.execute(
            "SELECT COUNT(*) FROM search_history WHERE user_id = ?", (user_id,)
        )
        count = c.fetchone()[0]
        if count >= 50:
            c.execute(
                "DELETE FROM search_history WHERE id IN "
                "(SELECT id FROM search_history WHERE user_id = ? ORDER BY searched_at ASC LIMIT ?)",
                (user_id, count - 49)
            )
        c.execute(
            "INSERT INTO search_history (user_id, query) VALUES (?, ?)",
            (user_id, query)
        )
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/search-history/<int:user_id>", methods=["GET"])
def get_search_history(user_id):
    """Return recent search queries for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c    = conn.cursor()
        c.execute(
            "SELECT query, searched_at FROM search_history "
            "WHERE user_id = ? ORDER BY searched_at DESC LIMIT 20",
            (user_id,)
        )
        history = [{"query": r[0], "searched_at": r[1]} for r in c.fetchall()]
        conn.close()
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_index():
    """Persist the in-memory local index to disk."""
    index_data = {
        "model_name": "all-MiniLM-L6-v2",
        "documents":  local_documents,
        "embeddings": local_embeddings
    }
    with open(LOCAL_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)


def _azure_upsert_doc(doc: dict):
    """Upload or update one document in Azure AI Search (best-effort)."""
    if not azure_configured:
        return
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        admin_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        )
        admin_client.upload_documents(documents=[{
            "@search.action": "mergeOrUpload",
            "id":       doc.get("id", ""),
            "content":  doc.get("text", ""),
            "category": doc.get("category", "General"),
            "snippet":  doc.get("snippet", "")
        }])
    except Exception as e:
        logger.warning("Azure upsert doc error: %s", e)


def _azure_delete_doc(doc_id: str):
    """Delete one document from Azure AI Search (best-effort)."""
    if not azure_configured:
        return
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        admin_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)
        )
        admin_client.delete_documents(documents=[{"@search.action": "delete", "id": doc_id}])
    except Exception as e:
        logger.warning("Azure delete doc error: %s", e)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nStarting SmartDoc Flask server on port 5000…")
    print(f"  ML Classifier:       {'ready' if model else 'NOT LOADED'}")
    print(f"  Local Semantic:      {'ready (%d docs)' % len(local_documents) if semantic_configured else 'not configured'}")
    print(f"  Azure AI Search:     {'ready (index: %s)' % AZURE_SEARCH_INDEX if azure_configured else 'not configured'}")
    print()
    app.run(port=5000, debug=True)
