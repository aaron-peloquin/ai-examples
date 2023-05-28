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
        original_input = inputs['input']
        inputs['input'] += ". (Clearly summarize the information you gather from Agents)"
        agent_output = self.agent.run(inputs)

        print(f"agent_output: ({agent_output})")
        conversation_inputs = {}
        conversation_inputs["human_input"] = original_input
        conversation_inputs["tool_output"] = agent_output

        conversation_output = self.conversation.run(conversation_inputs)
        print(f"conversation_output: ({conversation_output})")
        return {"text": conversation_output}
