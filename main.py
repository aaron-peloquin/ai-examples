import time
from colorama import Fore, Style

from langchain.agents import initialize_agent, AgentType
from langchain.chains import SimpleSequentialChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from modules.llm_vicuna7b import llm
from tools.DiceRoller import DiceRoller
from tools.dndSRD import dndSRD

# Set the memory to go back 4 turns
window_memory = ConversationBufferWindowMemory(k=4)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=window_memory,
)

conversation.prompt.template = '''You are an AI Assistant chatbot having a friendly conversation with a Human about Dungeons and Dragons.

If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
----

Human: {input}

AI:'''

# AI cannot roll dice on its own and should not make up random numbers, instead AI will use the "DiceRoller" Action to roll dice.
# When AI is unsure of the content and rules of Dungeons and Dragons (D&D), AI will use the "DNDSRD" Action to retrieve excerpts from the D&D rules book.

agent = initialize_agent(
    tools=[dndSRD(), DiceRoller()],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

from langchain.chains.base import Chain

from typing import Dict, List

class ConcatenateChain(Chain):
    chain_1: Chain
    chain_2: Chain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys)
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['text']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        agent_output = self.chain_1.run(inputs)
        print('=agent_output=', agent_output)
        print('=inputs1=', inputs)
        inputs['input'] += f"""

(Context: {agent_output})"""
        print('=inputs2=', inputs)
        conversation_output = self.chain_2.run(inputs)
        print('=conversation_output=', conversation_output)
        return {"text": conversation_output}
full_chain = ConcatenateChain(chain_1=agent, chain_2=conversation)

# dndQuestion = "Please roll 3d6 to determine my Charisma ability score"
# print(Fore.BLUE, f"Humanoid: {dndQuestion}", Style.RESET_ALL)
# agent_reply = agent.run(dndQuestion)
# print(Fore.LIGHTMAGENTA_EX, f"D&D Bot: {agent_reply}")

# dndQuestion = "(In Dungeons and Dragons) What age are Halflings considered mature?"
# print(Fore.BLUE, f"Humanoid: {dndQuestion}", Style.RESET_ALL)
# agent_reply = agent.run(dndQuestion)
# print(Fore.LIGHTMAGENTA_EX, f"D&D Bot: {agent_reply}")

print(Fore.RED, '====', Style.RESET_ALL, ' STARTING ', Fore.RED, '====')

while True:
    print(Fore.CYAN)
    human_input = input("Human:")
    start_time = time.time()
    print(Style.RESET_ALL)
    
    # ai_chat_reply = conversation.run(input=human_input)
    # ai_agent_reply = agent.run(human_input)
    ai_chat_agent_reply = full_chain.run(human_input)

    print(Fore.LIGHTMAGENTA_EX, ai_chat_agent_reply, Style.RESET_ALL)
    
    print(f"== {time.time() - start_time} ==")
