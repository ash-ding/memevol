import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import httpx
import jsonschema
import openai
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.spatial.distance import cosine

from common.logger import get_logger

log = get_logger("main")
load_dotenv()


# OpenAI rejects `reasoning_effort` for non-reasoning models. Keep a
# narrow allowlist by name prefix; update as new families appear.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_reasoning(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES)


# ---------------- Hire Agent ----------------
class Agent:
    """
    Asynchronous wrapper for the OpenAI Chat API.
    Allows custom system prompt, user prompt, and optional JSON schema validation.

    NOTE on `max_retries`: OpenAI's API has noticeable jitter — under
    moderate concurrency (a few hundred parallel calls), 10-20% of
    requests can hit transient timeouts or 5xx errors. We DEFAULT to
    `max_retries=5` for a reason: it converts almost all transient
    failures into eventual success. Harnesses that override to a smaller
    value (e.g. `max_retries=2`) silently drop work when the API is
    flaky — Phase 1 fact extraction returns empty, memory is incomplete,
    retrieval scores degrade. **Do not set `max_retries < 5` unless you
    have a specific reason and explicitly handle the failure path.**
    """
    def __init__(
        self,
        system_prompt: str,
        output_schema: Optional[Dict] = None,
        model: Optional[str] = 'gpt-4.1',
        timeout: float = 300.0,
        max_retries: int = 5,  # see class docstring — keep ≥5 to absorb API jitter
    ):
        self.model = model
        self.messages: List[Dict] = [{'role': 'system', 'content': system_prompt + (f"""ONLY output a valid JSON object conforming to the schema. The json schema is given below: {json.dumps(output_schema, ensure_ascii=False, indent=1)}""" if output_schema else "")}]
        self.output_schema = output_schema or None
        self.max_retries = max_retries
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
            max_retries=0,
        )

    def get_agent_config(self) -> Dict[str, Any]:
        """Return the current configuration for the agent."""
        config = {
            "model": self.model,
            "system_prompt": self.messages[0],
            "history": [msg for msg in self.messages[1:]]
        }
        return config

    async def ask(self, user_input, with_full_msg: bool = False, with_history: bool = False, temperature=None, reasoning_effort=None) -> Any:
        """
        Send a user message asynchronously and return the model's response.
        If an output schema is defined, the response will be parsed and validated.
        """
        self.messages.append({'role': 'user', 'content': user_input})
        if not with_history:
            chat_messages = [self.messages[0], self.messages[-1]]
        else:
            chat_messages = [self.messages[0]] + self.messages[1:]

        if with_full_msg:
            chat_messages = user_input

        kwargs = {
            'model': self.model,
            'messages': chat_messages
        }
        if self.output_schema:
            kwargs['response_format'] = {"type": "json_object"}
        if temperature:
            kwargs['temperature'] = temperature
        # Silently drop reasoning_effort when the backend model doesn't support it
        # (gpt-4*, non-reasoning OpenAI endpoints reject the field with a 400).
        if reasoning_effort and _supports_reasoning(self.model or ""):
            kwargs['reasoning_effort'] = reasoning_effort
        for app_attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                break
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
                if app_attempt == self.max_retries:
                    raise
                delay = 2 ** (app_attempt + 1)
                log.warning(f"[Agent] Retryable error (attempt {app_attempt+1}/{self.max_retries}): {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
            except Exception as e:
                print(e)
                raise

        from common.tokens import GLOBAL_TOKEN_TRACKER
        if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage"):
            await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)

        answer = resp.choices[0].message.content
        if not answer or answer.strip() == "":
            answer = "Error occur when generating answer."
            print('Error occur when generating answer.')

        self.messages.append({'role': 'assistant', 'content': answer})

        # --- Parse according to output schema if provided ---
        if self.output_schema:
            try:
                parsed = json.loads(answer)
                jsonschema.validate(instance=parsed, schema=self.output_schema)
                return parsed
            except json.JSONDecodeError:
                log.warning(f"fail to parse: {answer}")
                return {"error": "Model output is not valid JSON.", "raw_output": answer}
            except jsonschema.ValidationError as e:
                log.warning(f"fail to parse: {answer}")
                return {"error": f"Output does not match schema: {e.message}", "raw_output": answer}

        return answer


class Embedding:
    """
    Async embedding manager for computing single or batch embeddings,
    with optional similarity calculation. Supports token tracking.

    Batch chunking respects ALL three OpenAI embedding limits:
      • per-request items   <= 2048
      • per-request tokens  <= 290_000  (cap is 300k; we leave 10k margin)
      • per-input tokens    <= 8192     (text-embedding-3-*)
    Long single inputs are SILENTLY TRUNCATED to the per-input cap with a
    WARNING log line — common.llm.Embedding is the right place to absorb
    this rather than make every harness handle it.
    """

    # Per-process tiktoken encoder cache (cl100k_base is shared by all
    # text-embedding-3-* models, so populated lazily once per encoding).
    _encoder_cache: Dict[str, Any] = {}

    # Class constants for OpenAI embedding API limits.
    MAX_ITEMS_PER_REQUEST = 2048
    MAX_TOKENS_PER_REQUEST = 290_000   # 300k cap; 10k margin for tokenizer/server drift
    MAX_TOKENS_PER_INPUT = 8192        # text-embedding-3-* per-input cap

    def __init__(self, model: str = "text-embedding-3-small", retries: int = 3, retry_delay: float = 1.0):
        self.model = model
        self.retries = retries
        self.retry_delay = retry_delay

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(30.0, connect=10.0),
            max_retries=0,
        )

    def _get_encoder(self):
        """Lazy-load + cache a tiktoken encoder matched to the model."""
        if self.model not in self._encoder_cache:
            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback: cl100k_base covers all current text-embedding-3-* +
                # gpt-3.5/4 families; a safe default for unknown model strings.
                enc = tiktoken.get_encoding("cl100k_base")
            self._encoder_cache[self.model] = enc
        return self._encoder_cache[self.model]

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Safe synchronous entry point that works both inside and outside async loops."""
        if not texts:
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.get_batch_embeddings(texts))

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: asyncio.run(self.get_batch_embeddings(texts)))
            return future.result()

    async def get_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single text string asynchronously."""
        attempt = 0
        final_error = ''

        assert isinstance(text, str)

        while attempt < self.retries:
            try:
                resp = await self.client.embeddings.create(model=self.model, input=text)

                from common.tokens import GLOBAL_TOKEN_TRACKER
                if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage"):
                    await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)

                return resp.data[0].embedding
            except Exception as e:
                attempt += 1
                final_error = e
                if attempt >= self.retries:
                    break
                delay = self.retry_delay * (2 ** (attempt - 1))
                # Match Agent's retry log style: WARNING (not ERROR) since it's
                # a transient failure we're about to retry.
                log.warning(f"[Embedding] Retryable error (attempt {attempt}/{self.retries}): {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
        # Exhausted all retries → genuinely failed.
        log.error(f"[Embedding] Failed after {self.retries} attempts: {final_error}")
        raise RuntimeError(f"Failed to get embedding after {self.retries} attempts, with error: {final_error}")

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch, chunked by BOTH item count AND token budget.

        Why both: the OpenAI embedding API caps a single request at 2048 items
        AND ~300k tokens. Chunking only by item count fails on long inputs
        (e.g. 2000 dialogue turns can easily exceed 300k tokens). Per-input
        texts longer than 8192 tokens are silently truncated.
        """
        if not texts:
            return []

        enc = self._get_encoder()

        # Pre-tokenize once. Token IDs are reused for the actual API request
        # (sending pre-tokenized inputs guarantees the server sees exactly the
        # token count we budgeted against — no tokenizer/server drift).
        encoded: List[List[int]] = []
        truncated = 0
        for t in texts:
            ids = enc.encode(t or " ")  # empty string -> single space (safe; some servers reject [])
            if len(ids) > self.MAX_TOKENS_PER_INPUT:
                ids = ids[: self.MAX_TOKENS_PER_INPUT]
                truncated += 1
            encoded.append(ids)
        if truncated:
            log.warning(
                f"Embedding: truncated {truncated}/{len(texts)} input(s) to "
                f"{self.MAX_TOKENS_PER_INPUT} tokens (per-input cap)."
            )

        # Pack into chunks respecting both caps.
        chunks: List[List[int]] = []  # list of lists of indices into `texts`
        cur: List[int] = []
        cur_tok = 0
        for i, ids in enumerate(encoded):
            n = len(ids)
            if cur and (
                len(cur) >= self.MAX_ITEMS_PER_REQUEST
                or cur_tok + n > self.MAX_TOKENS_PER_REQUEST
            ):
                chunks.append(cur)
                cur = []
                cur_tok = 0
            cur.append(i)
            cur_tok += n
        if cur:
            chunks.append(cur)

        # Send chunks; preserve original order in output via index mapping.
        all_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_input = [encoded[i] for i in chunk]
            chunk_tok = sum(len(encoded[i]) for i in chunk)

            attempt = 0
            final_error: Any = None
            while attempt < self.retries:
                try:
                    resp = await self.client.embeddings.create(
                        model=self.model, input=chunk_input
                    )
                    from common.tokens import GLOBAL_TOKEN_TRACKER
                    if GLOBAL_TOKEN_TRACKER is not None and hasattr(resp, "usage"):
                        await GLOBAL_TOKEN_TRACKER.update(model_name=self.model, usage=resp.usage)
                    for j, item in zip(chunk, resp.data):
                        all_embeddings[j] = item.embedding
                    break
                except Exception as e:
                    attempt += 1
                    final_error = e
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
            else:
                log.error(
                    f"Embedding: chunk {chunk_idx + 1}/{len(chunks)} "
                    f"({len(chunk)} items, ~{chunk_tok} tokens) failed after "
                    f"{self.retries} attempts. Error: {final_error}"
                )
                raise RuntimeError(
                    f"Failed to get batch embeddings after {self.retries} attempts. With error: {final_error}"
                )

        return all_embeddings  # type: ignore[return-value]

    @staticmethod
    async def compute_similarity(emb1: List[float], emb2: List[float], metric: str = "cosine") -> float:
        """Asynchronously compute cosine similarity between two embeddings."""
        if metric == "cosine":
            return await asyncio.to_thread(lambda: 1 - cosine(emb1, emb2))
        else:
            log.error(f"Unsupported similarity metric: {metric}")
            raise ValueError(f"Unsupported similarity metric: {metric}")

    @staticmethod
    async def compute_one_to_group_similarity(
        emb: List[float],
        group_emb: List[List[float]],
        metric: str = "cosine"
    ) -> List[float]:
        """Asynchronously compute similarity between one embedding and a group of embeddings."""
        if not group_emb:
            return []
        tasks = [Embedding.compute_similarity(emb, g_emb, metric=metric) for g_emb in group_emb]
        return await asyncio.gather(*tasks)

    def embed_query(self, text: str) -> List[float]:
        """For Chroma compatibility: used when querying the collection."""
        return self([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """For Chroma compatibility: used when adding documents to the collection."""
        return self(texts)
