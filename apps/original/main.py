# `./library_loader.py` Enables importing from `/libs` directory (modulez)
import library_loader

import time
from colorama import Fore, Style

from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from modulez.chain_concatenate import ConcatenateChain
from modulez.callback_custom import CustomCallbackHandler
from llms.makersuite import MakerSuite
from tools.EquationSolver import EquationSolver

llm = MakerSuite()

from tools.DiceRoller import DiceRoller
from tools.dndSRD import dndSRD
from tools.DND5E import DND5E

handler = CustomCallbackHandler()

# Set the memory to go back 4 turns
window_memory = ConversationBufferWindowMemory(k=12)

conversation = ConversationChain(
    llm=llm,
    verbose=False,
    memory=window_memory,
)

conversation.prompt.template = '''AI Assistant chatbot having a friendly conversation with a Human about Dungeons and Dragons (D&D).
Assistant will act as the storyteller for a D&D game.
Helper will gather information, roll dice, and perform calculations instead for Assistant.
Assistant's job is to be the storyteller, not check rules or perform calculations.
Assistant will trust that Helper did its job completely and convey that information to the human.
If the Assistant does not know the answer to a question, it will not make up information.

Current conversation:
{history}
----
Human: {input}
Assistant: '''

agent = initialize_agent(
    tools=[
        DND5E(),
        DiceRoller(),
        EquationSolver(),
        dndSRD(),
    ],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

full_chain = ConcatenateChain(agent=agent, conversation=conversation, callbacks=[handler])

print(Fore.RED, '====', Style.RESET_ALL, ' STARTING ', Fore.RED, '====')

while True:
    print(Fore.CYAN)
    human_input = input("Human:")
    start_time = time.time()
    print(Style.RESET_ALL)

    ai_chat_agent_reply = full_chain.run(human_input)

    print(Fore.LIGHTMAGENTA_EX, ai_chat_agent_reply, Style.RESET_ALL)

    print(f"== {time.time() - start_time} ==")
