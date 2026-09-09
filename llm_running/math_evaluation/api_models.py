"""
API handler for closed-source models (OpenAI GPT, Anthropic Claude)
"""
import os
import time
import json
import tempfile
from typing import List, Optional, Dict, Any
from tqdm import tqdm


class APIModelHandler:
    """Base class for API-based model handlers"""
    
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 2048, 
                 top_p: float = 1.0, seed: Optional[int] = None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        
    def generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> List[str]:
        """Generate completions for a list of prompts"""
        raise NotImplementedError
        
    def generate_single(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate completion for a single prompt"""
        raise NotImplementedError


class OpenAIHandler(APIModelHandler):
    """Handler for OpenAI GPT models"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, 
                 temperature: float = 0.0, max_tokens: int = 2048, 
                 top_p: float = 1.0, seed: Optional[int] = None,
                 use_batch_api: bool = False,
                 batch_completion_window: str = "24h",
                 batch_poll_interval: float = 15.0):
        super().__init__(model_name, temperature, max_tokens, top_p, seed)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Please install it with: pip install openai"
            )
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it via api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.use_batch_api = use_batch_api
        self.batch_completion_window = batch_completion_window
        self.batch_poll_interval = max(batch_poll_interval, 1.0)

    def _build_chat_kwargs(self, prompt: str, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if stop:
            kwargs["stop"] = stop

        if self.seed is not None:
            kwargs["seed"] = self.seed

        return kwargs

    def _generate_via_batch_api(self, prompts: List[str], stop: Optional[List[str]] = None) -> List[str]:
        """Generate completions using OpenAI Batch API (/v1/chat/completions)."""
        if len(prompts) == 0:
            return []

        outputs = [""] * len(prompts)
        temp_input_file = None

        try:
            requests = []
            for idx, prompt in enumerate(prompts):
                requests.append(
                    {
                        "custom_id": str(idx),
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": self._build_chat_kwargs(prompt, stop),
                    }
                )

            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
                temp_input_file = tmp.name
                for item in requests:
                    tmp.write(json.dumps(item, ensure_ascii=False) + "\n")

            with open(temp_input_file, "rb") as fh:
                uploaded_file = self.client.files.create(file=fh, purpose="batch")

            batch = self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window=self.batch_completion_window,
            )

            print(f"Submitted OpenAI batch job: {batch.id}")
            last_status = None
            terminal_statuses = {"completed", "failed", "expired", "cancelled"}

            while True:
                batch = self.client.batches.retrieve(batch.id)
                status = batch.status
                if status != last_status:
                    print(f"OpenAI batch status: {status}")
                    last_status = status
                if status in terminal_statuses:
                    break
                time.sleep(self.batch_poll_interval)

            if batch.status != "completed":
                print(f"OpenAI batch did not complete successfully (status={batch.status}).")
                if getattr(batch, "error_file_id", None):
                    try:
                        error_payload = self.client.files.content(batch.error_file_id).content
                        if isinstance(error_payload, bytes):
                            error_text = error_payload.decode("utf-8", errors="replace")
                        else:
                            error_text = str(error_payload)
                        print("Batch error sample:")
                        print("\n".join(error_text.splitlines()[:20]))
                    except Exception as e:
                        print(f"Failed to fetch OpenAI batch error file: {e}")
                return outputs

            if not getattr(batch, "output_file_id", None):
                print("OpenAI batch completed but no output_file_id was returned.")
                return outputs

            content_payload = self.client.files.content(batch.output_file_id).content
            if isinstance(content_payload, bytes):
                lines = content_payload.decode("utf-8", errors="replace").splitlines()
            else:
                lines = str(content_payload).splitlines()

            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                custom_id = record.get("custom_id")
                if custom_id is None:
                    continue
                try:
                    idx = int(custom_id)
                except (TypeError, ValueError):
                    continue

                response_body = record.get("response", {}).get("body", {})
                text = ""
                choices = response_body.get("choices") or []
                if len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if isinstance(content, list):
                        text = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    else:
                        text = content or ""

                if 0 <= idx < len(outputs):
                    outputs[idx] = text

            return outputs

        except Exception as e:
            print(f"Error in OpenAI Batch API call: {e}")
            return outputs
        finally:
            if temp_input_file and os.path.exists(temp_input_file):
                try:
                    os.remove(temp_input_file)
                except OSError:
                    pass
        
    def generate_single(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate completion for a single prompt"""
        try:
            kwargs = self._build_chat_kwargs(prompt, stop)
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in OpenAI API call: {e}")
            return ""
    
    def generate(self, prompts: List[str], stop: Optional[List[str]] = None, 
                 batch_size: int = 10, delay: float = 0.5) -> List[str]:
        """Generate completions for a list of prompts with rate limiting"""
        if self.use_batch_api:
            return self._generate_via_batch_api(prompts, stop)

        outputs = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating with OpenAI"):
            batch = prompts[i:i + batch_size]
            batch_outputs = []
            
            for prompt in batch:
                output = self.generate_single(prompt, stop)
                batch_outputs.append(output)
                time.sleep(delay)  # Rate limiting
            
            outputs.extend(batch_outputs)
        
        return outputs


class AnthropicHandler(APIModelHandler):
    """Handler for Anthropic Claude models"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2048,
                 top_p: float = 1.0, seed: Optional[int] = None,
                 use_batch_api: bool = False,
                 batch_poll_interval: float = 15.0):
        super().__init__(model_name, temperature, max_tokens, top_p, seed)
        
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Please install it with: pip install anthropic"
            )
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Please provide it via api_key parameter "
                "or set ANTHROPIC_API_KEY environment variable."
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.use_batch_api = use_batch_api
        self.batch_poll_interval = batch_poll_interval
        
    def _generate_via_batch_api(self, prompts: List[str], stop: Optional[List[str]] = None) -> List[str]:
        """Generate completions using Anthropic's Message Batches API"""
        print(f"Using Anthropic Batch API for {len(prompts)} prompts...")
        outputs = [""] * len(prompts)
        
        if not prompts:
            return outputs
        
        try:
            # Build batch requests
            requests = []
            for idx, prompt in enumerate(prompts):
                requests.append({
                    "custom_id": str(idx),
                    "params": {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    }
                })
            
            # Create batch
            batch = self.client.messages.batches.create(requests=requests)
            print(f"Submitted Anthropic batch job: {batch.id}")
            
            # Poll for completion
            last_status = None
            terminal_statuses = {"ended"}
            
            while True:
                batch = self.client.messages.batches.retrieve(batch.id)
                status = batch.processing_status
                if status != last_status:
                    print(f"Anthropic batch status: {status}")
                    last_status = status
                if status in terminal_statuses:
                    break
                time.sleep(self.batch_poll_interval)
            
            # Check if completed successfully
            if batch.processing_status != "ended":
                print(f"Anthropic batch did not complete successfully (status={batch.processing_status}).")
                return outputs
            
            # Retrieve results
            results_decoder = self.client.messages.batches.results(batch.id)
            for result in results_decoder:
                custom_id = result.custom_id
                if custom_id is None:
                    continue
                try:
                    idx = int(custom_id)
                except (TypeError, ValueError):
                    continue
                
                # Extract text from result
                text = ""
                if hasattr(result.result, "message"):
                    message = result.result.message
                    if hasattr(message, "content") and len(message.content) > 0:
                        text = message.content[0].text
                elif hasattr(result.result, "error"):
                    print(f"Error in batch result {idx}: {result.result.error}")
                
                if 0 <= idx < len(outputs):
                    outputs[idx] = text
            
            print(f"Anthropic batch completed. Retrieved {sum(1 for o in outputs if o)} successful responses.")
            return outputs
        
        except Exception as e:
            print(f"Error in Anthropic Batch API: {e}")
            import traceback
            traceback.print_exc()
            return outputs
    
    def generate_single(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate completion for a single prompt"""
        try:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                #"top_p": self.top_p,
            }
            
            # Note: Anthropic doesn't support stop sequences in the same way
            # and doesn't support seed parameter for reproducibility
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
            
        except Exception as e:
            print(f"Error in Anthropic API call: {e}")
            return ""
    
    def generate(self, prompts: List[str], stop: Optional[List[str]] = None,
                 batch_size: int = 10, delay: float = 1.0) -> List[str]:
        """Generate completions for a list of prompts with rate limiting"""
        if self.use_batch_api:
            return self._generate_via_batch_api(prompts, stop)
        
        outputs = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating with Claude"):
            batch = prompts[i:i + batch_size]
            batch_outputs = []
            
            for prompt in batch:
                output = self.generate_single(prompt, stop)
                batch_outputs.append(output)
                time.sleep(delay)  # Rate limiting
            
            outputs.extend(batch_outputs)
        
        return outputs


def create_api_model_handler(model_name: str, temperature: float = 0.0, 
                            max_tokens: int = 2048, top_p: float = 1.0,
                            seed: Optional[int] = None, 
                            openai_api_key: Optional[str] = None,
                            anthropic_api_key: Optional[str] = None,
                            openai_use_batch: bool = False,
                            openai_batch_completion_window: str = "24h",
                            openai_batch_poll_interval: float = 15.0,
                            anthropic_use_batch: bool = False,
                            anthropic_batch_poll_interval: float = 15.0) -> APIModelHandler:
    """
    Factory function to create appropriate API model handler
    
    Args:
        model_name: Name of the model (e.g., 'gpt-4', 'gpt-3.5-turbo', 'claude-3-opus-20240229')
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        seed: Random seed for reproducibility (OpenAI only)
        openai_api_key: OpenAI API key (optional, can use env var)
        anthropic_api_key: Anthropic API key (optional, can use env var)
        openai_use_batch: Use OpenAI Batch API
        openai_batch_completion_window: OpenAI batch completion window
        openai_batch_poll_interval: OpenAI batch polling interval in seconds
        anthropic_use_batch: Use Anthropic Batch API
        anthropic_batch_poll_interval: Anthropic batch polling interval in seconds
    
    Returns:
        APIModelHandler: Appropriate handler for the model
    """
    model_lower = model_name.lower()
    
    # Detect OpenAI models
    if any(keyword in model_lower for keyword in ['gpt', 'o1', 'o3']):
        return OpenAIHandler(
            model_name=model_name,
            api_key=openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=seed,
            use_batch_api=openai_use_batch,
            batch_completion_window=openai_batch_completion_window,
            batch_poll_interval=openai_batch_poll_interval,
        )
    
    # Detect Anthropic Claude models
    elif 'claude' in model_lower:
        return AnthropicHandler(
            model_name=model_name,
            api_key=anthropic_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=seed,
            use_batch_api=anthropic_use_batch,
            batch_poll_interval=anthropic_batch_poll_interval,
        )
    
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Supported models are OpenAI (gpt-*) "
            f"and Anthropic (claude-*) models."
        )


def is_api_model(model_name: str) -> bool:
    """Check if a model name refers to an API-based model"""
    model_lower = model_name.lower()
    return any(keyword in model_lower for keyword in ['gpt', 'claude', 'o1', 'o3'])
