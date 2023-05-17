from typing import Any, List, Mapping, Optional
from dotenv import dotenv_values
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import re

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
        output_text = google_call.result
        if(stop is not None):
            output_text = self.truncate_string(output_text, stop)
        return output_text

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_output_tokens": self.max_output_tokens,
        }

    def truncate_string(self, string, stop_list):
        """
        Truncates a string as soon as it finds any of the items in the `stop_list`.

        Args:
            string: The string to truncate.
            stop_list: A list of strings that should be used to truncate the string.

        Returns:
            The truncated string.
        """

        pattern = re.compile("|".join(stop_list))
        match = pattern.search(string)
        if match:
            return string[:match.start()]
        return string