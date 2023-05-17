from typing import Any, List, Mapping, Optional
from dotenv import dotenv_values
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

import google.generativeai as palm

env_config = dotenv_values(".env")
palm.configure(api_key=env_config["MAKER_KEY"])

class MakerSuite(LLM):
    max_output_tokens: int = 800
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        google_call = palm.generate_text(
            model = "models/text-bison-001",
            prompt = prompt,
            max_output_tokens = self.max_output_tokens,
        )
        return google_call.result

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_output_tokens": self.max_output_tokens,
        }
