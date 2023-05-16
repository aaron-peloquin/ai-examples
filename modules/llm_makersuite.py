from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

import google.generativeai as palm

from MAKER_KEY import MAKER_KEY

palm.configure(api_key=MAKER_KEY())

class MakerSuite(LLM):
    max_output_tokens: int
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        print(f"stop: ({stop})")
        googleCall = palm.generate_text(
            model="models/text-bison-001",
            prompt=prompt,
            # The maximum length of the response
            max_output_tokens=800,
        )
        return googleCall.result

    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_output_tokens": self.max_output_tokens}
