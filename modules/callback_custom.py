from typing import Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.input import print_text

class CustomCallbackHandler(BaseCallbackHandler):
    """A custom callback handler
    
        Learning langchain
    """
    def __init__(self, color: Optional[str] = None) -> None:
        """Initialize callback handler."""
        self.color = color

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized["name"]
        print_text(f"\n\n\033[1m> Linking up a new [{class_name}] chain!?\033[0m")

    def on_llm_start(
        self,
        serialized: Dict[str,
        Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Print out the prompts."""

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        print(f'Callback ==output==: ({output})')
        print_text(f'Callback ==color==: ({color})')
        print_text(f'Callback ==observation_prefix==: ({observation_prefix})')
        print(f'Callback ==llm_prefix==: ({llm_prefix})')
        if observation_prefix is not None:
            print_text(f"\n{observation_prefix}")
        print_text(output, color=color if color else self.color)
        if llm_prefix is not None:
            print_text(f"\n{llm_prefix}")
