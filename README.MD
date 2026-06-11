# LLM Server

## Starting the Server

Submit the Slurm job:

```bash
sbatch llm.sh
```

Check the job logs to find the node's FQDN and confirm the server is up:

```bash
tail -f logs/log-<job_id>.out
# Reachable at: http://gpu-node-042.cluster.example.edu:9000
```

## API Endpoints

### Health check

```bash
curl http://<fqdn>:9000/health
```

```json
{"status": "healthy", "model": "Qwen/Qwen3.6-35B-A3B-FP8"}
```

### Single prompt — GET

```bash
curl "http://<fqdn>:9000/generate?prompt=What+is+the+capital+of+France%3F"
```

```json
{
  "prompt": "What is the capital of France?",
  "thinking": "The user is asking a simple geography question...",
  "response": "The capital of France is Paris.",
  "finish_reason": "stop",
  "output_tokens": 12,
  "temperature": 0.7,
  "max_tokens": 2048,
  "enable_thinking": true
}
```

Optional query parameters: `temperature` (0–2, default `0.7`), `max_tokens` (1–16384, default `2048`), `enable_thinking` (default `true`).

### Single prompt — POST

```bash
curl -X POST http://<fqdn>:9000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain backpropagation in one sentence.", "temperature": 0.3, "max_tokens": 256}'
```

```json
{
  "prompt": "Explain backpropagation in one sentence.",
  "thinking": "",
  "response": "Backpropagation computes gradients of the loss with respect to each parameter by applying the chain rule backwards through the network.",
  "finish_reason": "stop",
  "output_tokens": 28,
  "temperature": 0.3,
  "max_tokens": 256,
  "enable_thinking": false
}
```

### Batch prompts — POST

```bash
curl -X POST http://<fqdn>:9000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Summarize the Eiffel Tower.", "What year did WWII end?"],
    "temperature": 0.5,
    "max_tokens": 512,
    "enable_thinking": false
  }'
```

```json
{
  "prompts": ["Summarize the Eiffel Tower.", "What year did WWII end?"],
  "thinkings": ["", ""],
  "responses": ["The Eiffel Tower is a wrought-iron lattice tower in Paris...", "World War II ended in 1945."],
  "finish_reasons": ["stop", "stop"],
  "output_tokens": [42, 8],
  "temperature": 0.5,
  "max_tokens": 512,
  "enable_thinking": false
}
```

### Python client example

```python
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "http://<fqdn>:9000"

def generate(prompt, temperature=0.7, max_tokens=512, enable_thinking=True):
    resp = requests.post(f"{BASE}/generate", json={
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
    })
    resp.raise_for_status()
    return resp.json()

# Single prompt
result = generate("What is retrieval-augmented generation?")
print(result["response"])

# Many prompts in parallel via thread pool
prompts = ["Prompt A", "Prompt B", "Prompt C", ...]

with ThreadPoolExecutor(max_workers=16) as pool:
    futures = {pool.submit(generate, p, enable_thinking=False): p for p in prompts}
    for fut in as_completed(futures):
        data = fut.result()
        print(data["response"])
```

## Running with a Different Model

```bash
python3 llm.py --model_name leon-se/gemma-3-27b-it-FP8-Dynamic --port 9000
```

Supported models are passed directly to vLLM. Gemma models are detected automatically and use a plain chat template (no `<think>` block).

---

# Environment Set Up
Do the following in a cpu-allocated interactive job.

```
module add conda/latest
conda create -n myenv python=3.11 pip -y
conda activate myenv
python -m pip install vllm
```

Place your `HF_TOKEN` in `.env` after copying `.env.example`.

# Data Path
```
/scratch4/workspace/oyilmazel_umass_edu-stateful-retriever
/scratch4/workspace/oyilmazel_umass_edu-stateful-retriever/hf_cache/
/scratch4/workspace/oyilmazel_umass_edu-stateful-retriever/clean_trajectories/
```
