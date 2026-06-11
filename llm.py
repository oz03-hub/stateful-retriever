import os
import socket
import uuid
import asyncio
import argparse

from typing import Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = "/scratch4/workspace/oyilmazel_umass_edu-stateful-retriever/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = CACHE_PATH
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HF_DATASETS_CACHE"] = CACHE_PATH
os.environ["TORCH_HOME"] = CACHE_PATH
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

app = FastAPI()


class LLMServer:
    def __init__(self, model_name: str = "Qwen/Qwen3.6-35B-A3B-FP8"):
        self.model_name = model_name
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer = None
        self.is_gemma_model = "gemma" in model_name.lower()
        self.load_engine()
        self.load_tokenizer()

    def load_engine(self):
        print(f"Loading model: {self.model_name}...")

        engine_args = AsyncEngineArgs(
            model=self.model_name,
            download_dir=CACHE_PATH,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.92,
            max_model_len=64_512,
            max_num_seqs=512,
            max_num_batched_tokens=32_768,
            enable_prefix_caching=True,
            # quantization="compressed-tensors",  # fp8 (compute on A100 falls back via marlin kernels)
            trust_remote_code=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        print("Model loaded successfully")

    def load_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=CACHE_PATH,
        )

        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Tokenizer loaded successfully")

    def _format_prompt(self, p: str, enable_thinking: bool) -> str:
        messages = [{"role": "user", "content": p}]

        if self.is_gemma_model:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def _stop_tokens(self) -> List[str]:
        if self.is_gemma_model:
            return ["<end_of_turn>"]

        return ["<|eot_id|>", "<|im_end|>"]

    @staticmethod
    def _split_thinking(text: str, enable_thinking: bool) -> Tuple[str, str]:
        """Split raw model output into (thinking, response).
        Qwen3-style reasoning models emit `<think>...</think>` before the
        final answer. The chat template prepends `<think>` itself, so the
        generated text typically contains reasoning followed by `</think>`
        then the answer. If `</think>` is missing and thinking was on, the
        output was truncated inside the reasoning block — attribute it all
        to `thinking` rather than pretending it was the final answer.
        """

        if "</think>" not in text:
            if enable_thinking:
                return text.strip(), ""

            return "", text.strip()

        think_part, _, rest = text.partition("</think>")
        think_part = think_part.lstrip()

        if think_part.startswith("<think>"):
            think_part = think_part[len("<think>") :]

        return think_part.strip(), rest.strip()

    async def _generate_one(
        self, prompt: str, temperature: float, max_tokens: int, enable_thinking: bool
    ) -> Dict[str, str]:

        formatted = self._format_prompt(prompt, enable_thinking)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=self._stop_tokens(),
        )

        request_id = uuid.uuid4().hex

        final = None

        async for output in self.engine.generate(
            formatted, sampling_params, request_id
        ):
            final = output

        completion = final.outputs[0]

        text = completion.text.strip()

        thinking, response = self._split_thinking(text, enable_thinking)

        return {
            "thinking": thinking,
            "response": response,
            "finish_reason": completion.finish_reason or "",
            "output_tokens": len(completion.token_ids),
        }

    async def generate(
        self,
        prompt: Union[str, List[str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = True,
    ) -> Union[Dict[str, str], List[Dict[str, str]]]:

        is_single = isinstance(prompt, str)

        prompts = [prompt] if is_single else prompt

        results = await asyncio.gather(
            *[
                self._generate_one(p, temperature, max_tokens, enable_thinking)
                for p in prompts
            ]
        )

        return results[0] if is_single else results


llm_server: Optional[LLMServer] = None


def validate_parameters(temperature: float, max_tokens: int):

    if temperature < 0 or temperature > 2:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid temperature",
                "message": "Temperature must be between 0 and 2",
            },
        )

    if max_tokens < 1 or max_tokens > 16384:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid max_tokens",
                "message": "max_tokens must be between 1 and 16384",
            },
        )


class GenerateRequest(BaseModel):
    prompt: Optional[str] = None

    prompts: Optional[List[str]] = None

    temperature: float = 0.7

    max_tokens: int = 2048

    enable_thinking: bool = True


@app.get("/generate")
async def generate_get(
    prompt: str = Query("", description="Prompt string"),
    temperature: float = Query(0.7),
    max_tokens: int = Query(2048),
    enable_thinking: bool = Query(True),
):

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing prompt parameter",
                "message": 'Please provide a prompt via the "prompt" query parameter',
            },
        )

    validate_parameters(temperature, max_tokens)

    try:
        result = await llm_server.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Generation failed",
                "message": str(e),
            },
        )

    return {
        "prompt": prompt,
        "thinking": result["thinking"],
        "response": result["response"],
        "finish_reason": result["finish_reason"],
        "output_tokens": result["output_tokens"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
    }


@app.post("/generate")
async def generate_post(body: GenerateRequest):

    input_data: Union[str, List[str], None]

    input_data = body.prompts if body.prompts is not None else body.prompt

    if input_data is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing prompt",
                "message": 'Please provide either "prompt" (string) or "prompts" (list) in JSON body',
            },
        )

    validate_parameters(body.temperature, body.max_tokens)

    try:
        gen = await llm_server.generate(
            input_data,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            enable_thinking=body.enable_thinking,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Generation failed",
                "message": str(e),
            },
        )

    result = {
        "temperature": body.temperature,
        "max_tokens": body.max_tokens,
        "enable_thinking": body.enable_thinking,
    }

    if isinstance(input_data, list):
        result["prompts"] = input_data

        result["thinkings"] = [g["thinking"] for g in gen]

        result["responses"] = [g["response"] for g in gen]

        result["finish_reasons"] = [g["finish_reason"] for g in gen]

        result["output_tokens"] = [g["output_tokens"] for g in gen]

    else:
        result["prompt"] = input_data

        result["thinking"] = gen["thinking"]

        result["response"] = gen["response"]

        result["finish_reason"] = gen["finish_reason"]

        result["output_tokens"] = gen["output_tokens"]

    return result


@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "model": llm_server.model_name if llm_server else "not loaded",
    }


def main():

    global llm_server

    parser = argparse.ArgumentParser(
        description="LLM API Server using FastAPI and vLLM AsyncLLMEngine"
    )

    # leon-se/gemma-3-27b-it-FP8-Dynamic

    # Qwen/Qwen3-30B-A3B-Instruct-2507-FP8

    # Qwen/Qwen3.6-35B-A3B-FP8

    parser.add_argument(
        "--model_name", default="Qwen/Qwen3.6-35B-A3B-FP8", help="Model name to use"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port to bind the server to (default: 9000)",
    )

    args = parser.parse_args()

    print(f"Initializing LLM Server with model: {args.model_name}")

    llm_server = LLMServer(model_name=args.model_name)

    fqdn = socket.getfqdn()

    print(f"Hostname: {socket.gethostname()}")

    print(f"FQDN:     {fqdn}")

    print(f"Listening endpoint: http://{args.host}:{args.port}")

    print(f"Reachable at:       http://{fqdn}:{args.port}")

    print(f"Starting FastAPI server on {args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
