"""Batch 2 to 4 conversations (c15 to c40)."""
from __future__ import annotations

# c15: Poetry vs Pipenv [LONG: 22 msgs] (ChatGPT)
c15_msgs = [
    ("user", "hey im confused between poetry and pipenv for my new python backend service. which one handles lockfiles better?"),
    ("assistant", "Poetry is generally preferred today over Pipenv for several reasons:\n1. Lockfile resolution in Poetry uses an exact SAT solver (resolvelib-like algorithm) which is much faster and deterministic.\n2. Poetry adheres strictly to PEP 517/518 pyproject.toml standards.\n3. Pipenv historically had very slow locking issues on large dependency trees."),
    ("user", "wait what is the command in poetry to add a dev dependency like pytest"),
    ("assistant", "In modern Poetry (v1.2+), use dependency groups:\n```bash\npoetry add pytest --group dev\n```\nIn older versions it was `poetry add -D pytest`, but groups are the standard now."),
    ("user", "how does poetry export to requirements.txt for docker build"),
    ("assistant", "You use the poetry export plugin:\n```bash\npoetry export --without-hashes --format=requirements.txt > requirements.txt\n```\nYou can also add `--without dev` to only export production dependencies."),
    ("user", "wait why without-hashes? isn't hashes safer?"),
    ("assistant", "Hashes are safer for security, but pip install inside Docker often fails if platform wheels differ or if hashes for git/editable dependencies are missing. If you build linux wheels on mac, hashes might mismatch."),
    ("user", "can poetry resolve private pypi repositories?"),
    ("assistant", "Yes! You configure a secondary repository with:\n```bash\npoetry source add --priority=supplemental my-repo https://pypi.company.com/simple/\npoetry config http-basic.my-repo username password\n```"),
    ("user", "what if two dependencies need different versions of urllib3?"),
    ("assistant", "Poetry will fail fast with a SolverProblemError displaying the exact version conflict tree. Unlike pip which might silently overwrite or backtrack indefinitely, Poetry forces you to resolve the constraint."),
    ("user", "is there a poetry show command to inspect the tree?"),
    ("assistant", "Yes, run `poetry show --tree`. It prints the entire hierarchical dependency tree with active and requested versions."),
    ("user", "how to update only one package without updating everything?"),
    ("assistant", "Run `poetry update requests`. That updates only `requests` and its sub-dependencies, preserving the lockfile for all other packages."),
    ("user", "what about poetry run vs poetry shell?"),
    ("assistant", "`poetry run <cmd>` executes a command directly in the virtualenv without altering your current terminal environment. `poetry shell` spawns a subshell with the virtualenv activated."),
    ("user", "does poetry build create both sdist and wheel?"),
    ("assistant", "Yes, running `poetry build` outputs both a source distribution (`.tar.gz`) and a binary wheel (`.whl`) inside the `dist/` directory."),
    ("user", "awesome, this completely clarifies poetry lockfile resolution and workflow thanks!"),
    ("assistant", "You're welcome! Poetry's deterministic lockfile and pyproject.toml workflow make it a great choice for production python services.")
]

# c16: Conda environment export (Claude)
c16_msgs = [
    ("user", "How do I export a conda environment so colleagues on Mac and Linux can recreate it without build number errors?"),
    ("assistant", "Use `--no-builds` with conda env export:\n```bash\nconda env export --no-builds > environment.yml\n```\nEven better for cross-platform portability is `--from-history`:\n```bash\nconda env export --from-history > environment.yml\n```\nThis exports only the packages you explicitly installed rather than OS-specific sub-dependencies."),
    ("user", "What does conda env create -f environment.yml do if the env already exists?"),
    ("assistant", "It fails with `CondaValueError: prefix already exists`. To update an existing environment in place, run:\n```bash\nconda env update --name myenv --file environment.yml --prune\n```\nThe `--prune` flag removes packages that are no longer in the yaml file.")
]

# c17: uv cache clean and virtualenv activation (ChatGPT)
c17_msgs = [
    ("user", "my disk space is full because uv cached too many wheels. how do i clean it?"),
    ("assistant", "Run:\n```bash\nuv cache clean\n```\nYou can also run `uv cache prune` to remove only unreferenced or outdated pre-built wheels while keeping current cached packages."),
    ("user", "and how do i manually activate the venv uv created in bash?"),
    ("assistant", "By default, uv creates `.venv` in your project root. Activate it with:\n```bash\nsource .venv/bin/activate\n```\nOn Windows Command Prompt it's `.venv\\Scripts\\activate.bat` or `.venv\\Scripts\\Activate.ps1` in PowerShell.")
]

# c18: pyproject.toml build-system [DRIFT into cooking quick pasta] (Claude)
c18_msgs = [
    ("user", "what is the minimal pyproject.toml build-system section for flit or hatchling?"),
    ("assistant", "For Hatchling:\n```toml\n[build-system]\nrequires = [\"hatchling\"]\nbuild-backend = \"hatchling.build\"\n\n[project]\nname = \"mypackage\"\nversion = \"0.1.0\"\n```\nFor Flit:\n```toml\n[build-system]\nrequires = [\"flit_core >=3.2,<4\"]\nbuild-backend = \"flit_core.buildapi\"\n```"),
    ("user", "nice. while this build runs im starving. what is a super quick 10 min pasta i can make with pantry stuff?"),
    ("assistant", "Make Aglio e Olio! You just need pasta, olive oil, 3 cloves sliced garlic, chili flakes, and salt:\n1. Boil spaghetti in salted water.\n2. Gently sizzle sliced garlic and chili flakes in olive oil over low heat for 2 mins (don't burn garlic).\n3. Toss al dente pasta directly into the oil with 3 tbsp starchy pasta water to form a glossy emulsion. Serve immediately.")
]

# c19: PyTorch cuda vs mps (ChatGPT)
c19_msgs = [
    ("user", "how to write a universal device check in pytorch for cuda vs apple silicon mps vs cpu?"),
    ("assistant", "Use this standard pattern:\n```python\nimport torch\n\nif torch.cuda.is_available():\n    device = torch.device('cuda')\nelif torch.backends.mps.is_available():\n    device = torch.device('mps')\nelse:\n    device = torch.device('cpu')\nprint(f'Using device: {device}')\n```"),
    ("user", "does mps support all torch operations like complex numbers or scatter?"),
    ("assistant", "Most common operations are supported, but some specialized ops (e.g. certain 3D convolutions, complex fft, or double precision float64) may fall back to CPU or raise `NotImplementedError`. You can set `PYTORCH_ENABLE_MPS_FALLBACK=1` in your environment to automatically fall back to CPU for unsupported ops.")
]

# c20: CUDA OOM gradient checkpointing [DRIFT into cloud GPU costs] (Claude)
c20_msgs = [
    ("user", "My LLM fine-tuning job hits CUDA out of memory during backward pass on 24GB VRAM. Batch size is already 1!"),
    ("assistant", "Enable gradient checkpointing (activation checkpointing). Instead of storing all intermediate activations during the forward pass, it recalculates them during backward pass. In Hugging Face transformers:\n```python\nmodel.gradient_checkpointing_enable()\n```\nThis typically cuts memory by 40-60% at the cost of ~20% slower compute."),
    ("user", "That worked! But wow cloud A100 GPUs are expensive. I just saw my AWS bill is $450 this week. Any tips to not go broke?"),
    ("assistant", "Cloud GPU costs add up fast! Here are top strategies:\n1. Use Spot instances on AWS/GCP (often 60-70% discount) with auto-checkpointing.\n2. Rent on specialized GPU clouds like RunPod, Lambda Labs, or Vast.ai where A100s/H100s are often $1.50-$2.20/hr instead of $4+ on AWS.\n3. Set strict AWS budget alerts with auto-stop SNS lambdas so idle instances get shut down after 30 mins.")
]

# c21: fp16 vs bf16 mixed precision (ChatGPT)
c21_msgs = [
    ("user", "whats the difference between fp16 and bf16 in pytorch amp autocast?"),
    ("assistant", "The key difference is exponent vs mantissa bits:\n- `fp16`: 1 sign, 5 exponent, 10 mantissa bits. High precision but narrow dynamic range (max ~65504). Prone to underflow/overflow, requiring `GradScaler`.\n- `bf16`: 1 sign, 8 exponent, 7 mantissa bits. Same dynamic range as fp32 (8 exponent bits), but lower precision. It rarely underflows or overflows, so no GradScaler is needed.\nUse `torch.bfloat16` if your GPU supports it (NVIDIA Ampere A100/RTX 30xx or newer)."),
    ("user", "do I still use torch.cuda.amp.GradScaler with bf16?"),
    ("assistant", "No! With `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`, `GradScaler` is unnecessary and should be omitted because bf16 doesn't suffer from gradient underflow.")
]

# c22: DataLoader num_workers & pin_memory [LONG: 22 msgs] (Claude)
c22_msgs = [
    ("user", "My PyTorch GPU utilization is hovering around 30% during training. Profiler says it's waiting on DataLoader."),
    ("assistant", "Low GPU utilization with DataLoader waiting usually indicates CPU bottlenecking during data augmentation or host-to-device memory transfer. The two most important parameters to tune are `num_workers` and `pin_memory=True`."),
    ("user", "What should I set num_workers to?"),
    ("assistant", "A good rule of thumb is `num_workers = min(4 * num_gpus, os.cpu_count())`. Setting it too high causes memory bloat and context switching; setting it to 0 runs data loading in the main process synchronously."),
    ("user", "What does pin_memory=True actually do?"),
    ("assistant", "It allocates batch tensors in pinned (page-locked) host memory. This enables asynchronous DMA (Direct Memory Access) transfers from CPU RAM to GPU VRAM via PCIe, bypassing CPU paging."),
    ("user", "Should I also pass non_blocking=True to .to(device)?"),
    ("assistant", "Yes! `batch = batch.to(device, non_blocking=True)`. When paired with `pin_memory=True`, this lets data transfer happen concurrently with GPU computation from the previous batch."),
    ("user", "Does pin_memory work with custom collate_fn?"),
    ("assistant", "Yes, as long as your `collate_fn` returns standard PyTorch Tensors, dicts, or tuples of Tensors. If it returns custom non-tensor objects, PyTorch won't know how to pin them."),
    ("user", "What is persistent_workers=True?"),
    ("assistant", "By default, DataLoader worker processes are destroyed and respawned at each epoch boundary. `persistent_workers=True` keeps the worker pool alive across epochs, preventing repeated initialization overhead."),
    ("user", "What if I get a RuntimeError: DataLoader worker (pid ...) was killed by signal: Bus error?"),
    ("assistant", "Bus error in DataLoader almost always means Docker's shared memory (`/dev/shm`) is exhausted. Increase shared memory in Docker using `--shm-size=8g` or `--ipc=host`."),
    ("user", "Can prefetch_factor help too?"),
    ("assistant", "Yes, `prefetch_factor=2` tells each worker to pre-load 2 batches in advance so the queue is never empty when the GPU finishes a step."),
    ("user", "Is there any danger of memory leaks with persistent_workers?"),
    ("assistant", "If your dataset holds references to large objects in memory or opens file handles (like HDF5 or LMDB) without closing them, workers can slowly accumulate memory. Ensure datasets open handles lazily inside `__getitem__` per worker."),
    ("user", "This resolved my GPU bottleneck! GPU util is now 96%."),
    ("assistant", "Fantastic! Pairing `num_workers`, `pin_memory=True`, `non_blocking=True`, and `persistent_workers=True` is the gold standard for PyTorch data pipelines.")
]

# c23: Qdrant payload filtering (ChatGPT)
c23_msgs = [
    ("user", "how do i do compound filtering in qdrant with both must and should conditions in python"),
    ("assistant", "Use `models.Filter` from `qdrant_client.http`:\n```python\nfrom qdrant_client.http import models\n\nquery_filter = models.Filter(\n    must=[\n        models.FieldCondition(key=\"user_id\", match=models.MatchValue(value=\"usr_123\")),\n        models.FieldCondition(key=\"status\", match=models.MatchValue(value=\"active\")),\n    ],\n    should=[\n        models.FieldCondition(key=\"category\", match=models.MatchValue(value=\"ai\")),\n        models.FieldCondition(key=\"category\", match=models.MatchValue(value=\"python\")),\n    ]\n)\nclient.search(collection_name=\"docs\", query_vector=vec, query_filter=query_filter)\n```"),
    ("user", "does qdrant support range filtering on numeric timestamps?"),
    ("assistant", "Yes, use `models.Range`:\n```python\nmodels.FieldCondition(key=\"created_at\", range=models.Range(gte=1672531200, lte=1704067200))\n```\nMake sure to create a payload index on `created_at` with `schema=models.PayloadSchemaType.INTEGER` for fast lookups.")
]

# c24: Hybrid search BM25 and Qdrant fusion (Claude)
c24_msgs = [
    ("user", "How can I combine lexical BM25 results from SQLite FTS5 with dense vector results from Qdrant?"),
    ("assistant", "The standard and robust method is Reciprocal Rank Fusion (RRF):\n1. Run the query against SQLite FTS5 to get top K BM25 results.\n2. Run the query embedding against Qdrant to get top K dense results.\n3. Take the ranked list from each (sorted best-first). Discard the raw scores.\n4. For each candidate chunk, compute RRF score = sum(1.0 / (k + rank)) across sources, with standard k=60.\n5. Sort by RRF score descending. This avoids score calibration problems between BM25 and cosine similarity.")
]

# c25: Cross-encoder reranking [LONG: 22 msgs] (ChatGPT)
c25_msgs = [
    ("user", "why should i use a cross-encoder to rerank after rrf fusion in rag?"),
    ("assistant", "Bi-encoders (like vector embeddings) compute query and document representations independently in vector space. While very fast for first-stage candidate retrieval, they miss token-level cross-attention interactions.\nA cross-encoder passes the query and document together into the transformer layers simultaneously (`[CLS] query [SEP] document [SEP]`), allowing full self-attention across every word pair. This provides drastically superior relevance scoring."),
    ("user", "which cross-encoder model is recommended for general text search?"),
    ("assistant", "`BAAI/bge-reranker-base` or `BAAI/bge-reranker-large` are currently state of the art open weights models. `bge-reranker-base` runs fast on CPU or small GPUs with sentence-transformers."),
    ("user", "how do i load and predict with BAAI/bge-reranker-base in python?"),
    ("assistant", "```python\nfrom sentence_transformers import CrossEncoder\n\nmodel = CrossEncoder('BAAI/bge-reranker-base')\npairs = [[query, doc_text] for doc_text in candidates]\nscores = model.predict(pairs, batch_size=32)\n```"),
    ("user", "are the scores raw logits or probabilities?"),
    ("assistant", "By default, `model.predict(pairs)` returns raw unnormalized logits (which can range from negative numbers like -8.0 up to positive +8.0). If you need scores bounded in [0, 1], apply `scipy.special.expit(scores)` (sigmoid) or pass `apply_softmax=True` if supported."),
    ("user", "how many candidates should i pass into reranking?"),
    ("assistant", "Typically 20 to 50 candidates from first-stage retrieval. Passing 100+ candidates adds excessive latency since cross-encoders are $O(N)$ transformer inferences per query."),
    ("user", "what latency should i expect on cpu vs gpu for 30 candidates?"),
    ("assistant", "On an modern 8-core CPU, scoring 30 candidates with bge-reranker-base takes ~150-300ms. On an NVIDIA GPU (T4 or A10), it takes ~15-25ms."),
    ("user", "can a cross-encoder score be used for relevance gating?"),
    ("assistant", "Yes! Because cross-encoder scores measure true semantic relevance rather than relative rank, setting a minimum score threshold (like sigmoid score >= 0.35 or logit >= -2.0) is very effective at filtering out irrelevant noise."),
    ("user", "should query and doc be truncated if too long?"),
    ("assistant", "Yes, `CrossEncoder(..., max_length=512)`. Excessively long documents will be truncated; in RAG, you want each candidate chunk to be around 200-400 tokens."),
    ("user", "does bge-reranker support multilingual text?"),
    ("assistant", "Yes, `bge-reranker-base` and `bge-reranker-large` support both English and Chinese, with solid performance across other major languages."),
    ("user", "awesome, this covers the cross-encoder reranker architecture completely."),
    ("assistant", "Glad to help! A hybrid BM25 + dense retrieval pipeline with RRF followed by a BGE cross-encoder reranker is industry standard.")
]

# c26: Pandas merge vs join (Claude)
c26_msgs = [
    ("user", "What is the practical difference between pandas DataFrame.merge and DataFrame.join?"),
    ("assistant", "`merge` is a module-level or DataFrame method that joins on columns (or index) using database-style join syntax (`on=['col1', 'col2']`, `how='inner'|'left'|'right'|'outer'`).\n`join` is a convenience DataFrame method specifically designed to join on indexes (or a column against another DataFrame's index).\nIn almost all multi-column or complex join scenarios, `df.merge()` is clearer and more explicit.")
]

# c27: Polars LazyFrame filter pushdown [DRIFT into ergonomics/back pain] (ChatGPT)
c27_msgs = [
    ("user", "how does polars lazyframe optimize queries compared to eager dataframe?"),
    ("assistant", "When using `pl.scan_parquet()` or `df.lazy()`, Polars builds a logical query plan without executing immediately. The query optimizer applies:\n1. Predicate pushdown: filters (`.filter()`) are evaluated at the storage level, skipping unneeded parquet row groups.\n2. Projection pushdown: only referenced columns (`.select()`) are loaded into RAM.\n3. Common subplan elimination and slice pushdown.\nCall `.collect()` to execute the optimized plan."),
    ("user", "my back is killing me after sitting debugging all day. any advice on chair or desk setup?"),
    ("assistant", "Back pain from desk work is brutal! Quick checklist:\n1. Screen height: top third of monitor should be at eye level so your neck isn't bent forward.\n2. Elbows at 90 degrees resting naturally on armrests, shoulders relaxed.\n3. Lower back support: place a small rolled towel behind your lumbar spine if your chair lacks support.\n4. Take a 2-minute walking break every 45 minutes.")
]

# c28: SQLite FTS5 bm25 query syntax (Claude)
c28_msgs = [
    ("user", "How do I query an SQLite FTS5 virtual table using the built-in bm25 ranking function?"),
    ("assistant", "Use `MATCH` and the `bm25(table_name)` auxiliary function in `ORDER BY`:\n```sql\nSELECT chunk_id, bm25(chunks_fts) AS rank_score\nFROM chunks_fts\nWHERE chunks_fts MATCH 'python AND virtualenv'\nORDER BY rank_score ASC\nLIMIT 10;\n```\nNote that in SQLite FTS5, `bm25()` returns negative numbers where more negative (lower) values mean higher relevance! So you sort `ASC` for best-first.")
]

# c29: DuckDB querying partitioned parquet (ChatGPT)
c29_msgs = [
    ("user", "can duckdb query a partitioned hive-style parquet folder directly with glob patterns?"),
    ("assistant", "Yes! DuckDB handles Hive partitioning out of the box:\n```sql\nSELECT year, category, count(*), avg(amount)\nFROM read_parquet('data/sales/*/*/*.parquet', hive_partitioning=true)\nWHERE year = 2023\nGROUP BY year, category;\n```\nDuckDB automatically pushes down partition filters to skip irrelevant partition folders without reading files.")
]

# c30: FastAPI background tasks vs Celery (Claude)
c30_msgs = [
    ("user", "When should I use FastAPI BackgroundTasks vs a dedicated Celery worker with Redis?"),
    ("assistant", "Use `FastAPI.BackgroundTasks` when:\n- The task is short and lightweight (sending an email confirmation, writing a log file, firing a webhook).\n- Losing a task on server restart/crash is acceptable.\n- You want zero external infrastructure.\n\nUse `Celery` (or Dramatiq/Arq) when:\n- Tasks are long-running (video encoding, batch data processing, ML inference).\n- You need persistence, retries with exponential backoff, rate limiting, and dedicated worker scaling across multiple machines.")
]

# c31: Pydantic v2 migration [LONG: 22 msgs] (ChatGPT)
c31_msgs = [
    ("user", "I'm migrating a FastAPI codebase from Pydantic v1 to Pydantic v2. What replaced @validator and @root_validator?"),
    ("assistant", "In Pydantic v2:\n- `@validator` is replaced by `@field_validator('field_name', mode='before'|'after')`.\n- `@root_validator` is replaced by `@model_validator(mode='before'|'after')`.\nAlso, `mode='after'` validators receive the validated model instance `self` directly in model_validator, whereas `mode='before'` validators receive the raw dictionary `data` before field validation."),
    ("user", "How do I forbid extra fields in a Pydantic v2 model?"),
    ("assistant", "Instead of the old `class Config:` inner class, use `model_config = ConfigDict(extra='forbid')`:\n```python\nfrom pydantic import BaseModel, ConfigDict\n\nclass MyModel(BaseModel):\n    model_config = ConfigDict(extra='forbid')\n    name: str\n    age: int\n```"),
    ("user", "What happened to .dict() and .json()?"),
    ("assistant", "They are deprecated and will be removed in v3. Use:\n- `model.model_dump()` instead of `.dict()`\n- `model.model_dump_json()` instead of `.json()`"),
    ("user", "What replaced parse_obj()?"),
    ("assistant", "Use `MyModel.model_validate(obj)` instead of `MyModel.parse_obj(obj)`. For JSON strings, use `MyModel.model_validate_json(raw_json)`."),
    ("user", "Can you show an example of @model_validator(mode='after') checking cross-field constraints?"),
    ("assistant", "```python\nfrom typing_extensions import Self\nfrom pydantic import BaseModel, model_validator\n\nclass DateRange(BaseModel):\n    start_date: int\n    end_date: int\n\n    @model_validator(mode='after')\n    def check_dates(self) -> Self:\n        if self.start_date > self.end_date:\n            raise ValueError('start_date cannot be greater than end_date')\n        return self\n```"),
    ("user", "What about schema() for OpenAPI generation?"),
    ("assistant", "Use `MyModel.model_json_schema()` instead of `MyModel.schema()`."),
    ("user", "Is Pydantic v2 much faster because of pydantic-core?"),
    ("assistant", "Yes, pydantic-core is written in Rust. Parsing and validation benchmarks typically show 5x to 20x speedups compared to v1 pure Python implementation."),
    ("user", "Does Field(..., regex=...) still work?"),
    ("assistant", "No, `regex` was renamed to `pattern`, e.g. `Field(pattern=r'^[a-z]+$')`."),
    ("user", "What about custom types with __get_validators__?"),
    ("assistant", "Custom types now use `__get_pydantic_core_schema__` from `pydantic_core.core_schema` or `Annotated[T, BeforeValidator(...)]`."),
    ("user", "This covers all the breaking changes I was hitting, thanks!"),
    ("assistant", "Glad to help! The v1 to v2 migration is straightforward once you know the new `model_dump`, `ConfigDict`, and validator conventions.")
]

# c32: Django ORM select_related vs prefetch_related (Claude)
c32_msgs = [
    ("user", "In Django ORM, when should I use select_related vs prefetch_related?"),
    ("assistant", "- `select_related`: Performs an SQL `JOIN` in a single query. Use for single-valued relationships (ForeignKey and OneToOneField).\n- `prefetch_related`: Executes separate SQL queries and performs joining in Python. Use for multi-valued relationships (ManyToManyField and reverse ForeignKey).\nUsing `select_related` on ManyToMany will raise a FieldError.")
]

# c33: JWT Bearer auth [DRIFT into vacation planning] (ChatGPT)
c33_msgs = [
    ("user", "how to extract and verify a Bearer JWT in a fastapi header dependency?"),
    ("assistant", "Use `HTTPBearer` from `fastapi.security` and `jwt.decode` from PyJWT:\n```python\nfrom fastapi import Depends, HTTPException, status\nfrom fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\nimport jwt\n\nsecurity = HTTPBearer()\n\ndef get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):\n    token = creds.credentials\n    try:\n        payload = jwt.decode(token, 'SECRET_KEY', algorithms=['HS256'])\n        return payload['sub']\n    except jwt.PyJWTError:\n        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token')\n```"),
    ("user", "finally got this auth working. once this sprint is done im taking a week off. where is a chill beach place to relax?"),
    ("assistant", "Congrats on finishing the auth! For a chill beach getaway:\n- Goa (South Goa like Palolem or Agonda) is quiet, scenic, and great for unwinding without loud crowds.\n- Andaman Islands (Havelock / Radhanagar Beach) has turquoise waters and amazing diving.\n- If Southeast Asia: Da Nang / Hoi An in Vietnam has great beaches and delicious food on a budget.")
]

# c34: git cherry-pick conflicts (Claude)
c34_msgs = [
    ("user", "I cherry-picked a commit into release branch and got merge conflicts. How do I finish it?"),
    ("assistant", "1. Open the conflicted files, resolve the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`), and save.\n2. Stage the resolved files with `git add <file>`.\n3. Continue the cherry-pick with:\n```bash\ngit cherry-pick --continue\n```\nIf you want to abort and return to the state before cherry-picking, run `git cherry-pick --abort`.")
]

# c35: git stash pop vs apply (ChatGPT)
c35_msgs = [
    ("user", "whats the difference between git stash pop and git stash apply?"),
    ("assistant", "- `git stash pop`: Applies the most recent stashed changes (`stash@{0}`) to your working directory AND removes it from the stash list if there are no conflicts.\n- `git stash apply`: Applies the changes to your working directory but KEEPS the stash entry in your stash list so you can re-use it elsewhere.")
]

# c36: git reflog recover deleted branch (Claude)
c36_msgs = [
    ("user", "I accidentally deleted a local branch with `git branch -D feature-login` before pushing! Is it gone forever?"),
    ("assistant", "No, it's not gone! Git keeps commits in the reflog for typically 30-90 days:\n1. Run `git reflog`.\n2. Look for the last commit hash where you worked on `feature-login` (e.g. `abc1234 HEAD@{2}: commit: fix login validation`).\n3. Recreate the branch from that commit:\n```bash\ngit checkout -b feature-login abc1234\n```")
]

# c37: git submodule update [DRIFT into fixing squeaky door latch] (ChatGPT)
c37_msgs = [
    ("user", "cloned a repo with git submodules and the subfolder is empty. what command populates it?"),
    ("assistant", "Run:\n```bash\ngit submodule update --init --recursive\n```\nAlternatively, when cloning a new repository, pass the `--recurse-submodules` flag:\n```bash\ngit clone --recurse-submodules <repo-url>\n```"),
    ("user", "that worked. totally random question, my bedroom door latch squeaks every time i close it. how to fix?"),
    ("assistant", "Spray a tiny squirt of WD-40 or silicone lubricant directly onto the spring latch bolt and the hinge pins, then open and close the door 10 times to let it penetrate. If you don't have spray lube, rub a little petroleum jelly (Vaseline) or pencil graphite on the latch face!")
]

# c38: React useEffect cleanup function (Claude)
c38_msgs = [
    ("user", "Why does my React component keep receiving WebSocket messages after the user navigates away?"),
    ("assistant", "You need to return a cleanup function from your `useEffect` hook to close the WebSocket connection when the component unmounts:\n```javascript\nuseEffect(() => {\n  const ws = new WebSocket('wss://api.example.com/stream');\n  ws.onmessage = (event) => setData(JSON.parse(event.data));\n\n  return () => {\n    ws.close();\n  };\n}, []);\n```\nWithout the cleanup return function, the connection remains open in memory.")
]

# c39: React useState stale closure in setInterval (ChatGPT)
c39_msgs = [
    ("user", "my count state stays at 1 inside setInterval even though i call setCount(count + 1). why?"),
    ("assistant", "This is a classic 'stale closure' bug. The interval callback captures the `count` variable from the initial render when it was 0.\nTo fix it, use the functional state updater form:\n```javascript\nuseEffect(() => {\n  const id = setInterval(() => {\n    setCount(prev => prev + 1);\n  }, 1000);\n  return () => clearInterval(id);\n}, []);\n```\nThe updater function always receives the fresh current state.")
]

# c40: Next.js server vs client components (Claude)
c40_msgs = [
    ("user", "In Next.js App Router, when must a component have 'use client' at the top?"),
    ("assistant", "You only need `'use client'` when the component uses:\n1. React state and lifecycle hooks (`useState`, `useEffect`, `useReducer`).\n2. Browser APIs (`window`, `localStorage`, `navigator`).\n3. DOM event listeners (`onClick`, `onChange`, `onSubmit`).\n4. Custom hooks that depend on client features.\nEverything else (fetching data from DB, reading cookies, rendering static markdown) should remain a Server Component for minimal JavaScript bundle size.")
]

