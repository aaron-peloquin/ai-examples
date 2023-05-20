import time
from colorama import Fore, Style

from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from modules.chain_concatenate import ConcatenateChain
from modules.llm_makersuite import MakerSuite

llm = MakerSuite()

from tools.DiceRoller import DiceRoller
from tools.dndSRD import dndSRD
from tools.DND5E import DND5E

# Set the memory to go back 4 turns
window_memory = ConversationBufferWindowMemory(k=12)

conversation = ConversationChain(
    llm=llm,
    verbose=False,
    memory=window_memory,
)
conversation.prompt.template = '''AI Assistant chatbot having a friendly conversation with a Human about Dungeons and Dragons.

If the Assistant does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
----
Human: {input}
Assistant: '''

agent = initialize_agent(
    tools=[DND5E(), DiceRoller(), dndSRD()],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

full_chain = ConcatenateChain(agent=agent, conversation=conversation)

print(Fore.RED, '====', Style.RESET_ALL, ' STARTING ', Fore.RED, '====')

while True:
    print(Fore.CYAN)
    human_input = input("Human:")
    start_time = time.time()
    print(Style.RESET_ALL)
    
    ai_chat_agent_reply = full_chain.run(human_input)

    print(Fore.LIGHTMAGENTA_EX, ai_chat_agent_reply, Style.RESET_ALL)
    
    print(f"== {time.time() - start_time} ==")
