from langchain.chains.base import Chain

from typing import Dict, List

class ConcatenateChain(Chain):
    agent: Chain
    conversation: Chain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.agent.input_keys)
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['text']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        agent_output = self.agent.run(inputs)
        inputs['input'] += f"""

(Helper: {agent_output})"""
        conversation_output = self.conversation.run(inputs)
        return {"text": conversation_output}
