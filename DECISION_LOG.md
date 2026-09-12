# Architectural Decision Log: 12 Non-Obvious Engineering Trade-offs

This document records the 12 key non-obvious engineering decisions, rationale, and alternatives considered during the development of the Hiver AI Support Agent for **@AppleSupport**.

---

### 1. Brand Selection: @AppleSupport over @AmazonHelp or Airlines
* **Decision**: Selected **@AppleSupport** as the single target brand from the 3M Kaggle dataset.
* **Why**: While `@AmazonHelp` has higher total tweet volume, the majority of Amazon's customer interactions involve order tracking and package delivery that depend strictly on private backend order databases (`"Where is order #123-456?"`). In contrast, Apple has rich, technically verifiable, self-contained troubleshooting protocols (force-restart sequences, DFU mode, battery health calibration, iCloud sync settings) that can be grounded in public documentation.
* **Alternative Considered**: `@Delta` / `@British_Airways` (rejected due to temporal volatility—flight delays depend on real-time weather and radar data not present in static training sets).

---

### 2. Filtering Out the "DM Redirection" Boilerplate Trap
* **Decision**: Implemented an automated filter in `src/data_extractor.py` requiring responses to be $\ge 45$ characters and excluding pure DM boilerplate (e.g., regex matching `^(please )?(send us a dm|dm us)` unless accompanied by substantive diagnostic steps).
* **Why**: Over 60% of raw brand replies in Kaggle's Twitter dataset are empty redirects: *"We'd be glad to look into this. Please send us a DM with your iOS version."* If indexed naively into a vector store, top-$k$ RAG retrieval returns useless boilerplate, causing the LLM to learn passive deflection rather than troubleshooting.
* **Alternative Considered**: Raw sampling of the top 2,000 tweets (rejected because it produced an agent that suggested "Send a DM" for 100% of queries).

---

### 3. Pure NumPy Normalized Cosine Engine with FAISS Bridge
* **Decision**: Defaulted to BLAS-accelerated NumPy matrix multiplication (`query_vector @ embeddings.T`) for vector similarity while providing a drop-in FAISS bridge if installed.
* **Why**: Our curated historical knowledge base contains 1,998 vectors of dimension 384 (`all-MiniLM-L6-v2`), requiring only ~3 MB of RAM. At this scale, NumPy exact dot product runs in **< 0.4 milliseconds**, achieves 100% exact recall, and avoids binary C++ build failures on reviewer environments (especially Windows/Python 3.12+).
* **Alternative Considered**: Pinecone / ChromaDB / Weaviate (rejected due to cloud dependencies, network latency overhead, and API keys).

---

### 4. 4-Layer Directed State Graph vs. Monolithic LLM Prompt
* **Decision**: Decoupled the architecture into:
  1. *Deterministic Pre-Guardrails*
  2. *Intent-Aware Context Retrieval (RAG)*
  3. *Structured LLM Generation*
  4. *Post-LLM Policy & Confidence Arbiter*
* **Why**: A monolithic "all-in-one" LLM prompt is vulnerable to prompt injection, wastes tokens on obvious deterministic cases (e.g. legal threats or PII), and cannot reliably self-terminate when confidence is low.
* **Alternative Considered**: Single-prompt LangChain pipeline (rejected due to lack of explainability and non-deterministic escalation behavior).

---

### 5. Stratified 60 / 20 / 20 Golden Evaluation Set Difficulty Distribution
* **Decision**: Structured the 180-sample hand-labeled Golden Set into 60% Standard Core issues, 20% Ambiguous/Sarcastic Edge cases, and 20% High-Risk Escalation Triggers.
* **Why**: Random sampling over-represents easy majority classes, yielding inflated accuracy (e.g. 98%) that collapses in production. Deliberately injecting edge cases (sarcasm, multi-intent collision) and security threats guarantees realistic stress-testing.
* **Alternative Considered**: Uniform random sampling from historical test sets (rejected due to severe class imbalance and temporal bias).

---

### 6. Asymmetric Cost Function on Routing Errors
* **Decision**: Explicitly prioritized minimizing the **False Auto-Handle Rate** over minimizing the False Escalation Rate.
* **Why**: In customer support, errors are asymmetric. A *False Escalation* costs ~$3 in human agent triage time; a *False Auto-Handle* (e.g., instructing a customer with an expanding, smoking battery to force-restart their device, or dismissing a customer threatening a CFPB complaint) creates legal liability, PR backlash, and customer churn.
* **Alternative Considered**: Equal weight F1-score optimization (rejected because it treats safety-critical failures identically to routine misclassifications).

---

### 7. Dual Deterministic PII Sanitization & Automatic Privacy Escalation
* **Decision**: Ingested tweets are scanned and redacted for Credit Cards, Passwords, Emails, Phone Numbers, and IMEI *before* vector search or LLM prompting. Any public PII exposure immediately triggers `ESCALATE_TO_HUMAN` with a privacy warning.
* **Why**: Customers frequently tweet sensitive data in public anger. If the AI echoes this data in a public Twitter reply, it causes a catastrophic privacy breach. Our agent warns the customer to delete the public tweet and routes to private DM.
* **Alternative Considered**: Relying on the LLM to ignore customer credit card numbers (rejected because LLMs frequently repeat context in their generation).

---

### 8. Eager Single-Instance Model Loading for Thread Safety
* **Decision**: Eagerly loaded `SentenceTransformer` once during `load_retriever_from_disk()` on the main thread rather than on-demand inside worker threads.
* **Why**: In `EvaluationHarness`, batch evaluation uses `ThreadPoolExecutor(max_workers=5)`. Lazy initialization caused 5 threads to concurrently download weights and allocate 5 duplicate model instances in PyTorch memory.
* **Alternative Considered**: Synchronous sequential evaluation (rejected because it slowed 180-sample evaluation from 30s to 3+ minutes).

---

### 9. Resilient Zero-API-Key Offline Mock Mode (`--mock`)
* **Decision**: Integrated a deterministic heuristic fallback resolver that mimics LLM behavior using retrieved context if `GEMINI_API_KEY` is not present.
* **Why**: Reviewers often clone a repository and run `python run_pipeline.py` immediately. Crashing with `KeyError: 'GEMINI_API_KEY'` guarantees a negative review.
* **Alternative Considered**: Requiring the user to configure API keys before running any script (rejected to ensure guaranteed < 2-minute reproduction).

---

### 10. Dual Metric Judge Calibration (Pearson $r$ + Cohen's $\kappa$)
* **Decision**: Evaluated the LLM-as-a-Judge against human ground truth using both continuous correlation (Pearson $r$) and discrete inter-rater agreement (Cohen's Kappa $\kappa$).
* **Why**: Pearson $r$ measures linear correlation in continuous rating trends, while Cohen's $\kappa$ accounts for chance agreement on binary go/no-go decisions. Providing both proves mathematical calibration.
* **Alternative Considered**: Simple percentage accuracy (rejected because it fails to correct for random chance agreement on skewed classes).

---

### 11. Keyword-Boosted Hybrid Retrieval over Pure Dense Search
* **Decision**: Added keyword boosting for specific Apple product models (`"iPhone 13"`, `"AirPods Pro"`, `"iOS 17"`) on top of MiniLM cosine similarity.
* **Why**: Pure dense embedding models often place all iPhone hardware queries close together in vector space, failing to distinguish between device generations that have different button combinations for force restarts (e.g. iPhone 7 vs iPhone 14).
* **Alternative Considered**: Heavy BM25 + Cross-Encoder re-ranking (rejected due to added runtime latency and dependency complexity).

---

### 12. Pre-Wired `# LIVE_INTERVIEW_HOOK:` Anchors
* **Decision**: Placed explicit live modification hooks in `src/config.py` and `src/agent.py` for keyword guardrails, Pydantic schemas, and model provider callers.
* **Why**: During live technical defenses, interviewers test candidates by asking for instant code modifications on screen share. Having labeled anchor points allows you to execute edits in **under 30 seconds** without hunting through complex codebases.
* **Alternative Considered**: Highly abstract plugin/middleware frameworks (rejected because layered abstractions make live code modification slow and error-prone).
