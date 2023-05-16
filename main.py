import time
from colorama import Fore, Style

from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from modules.ConcatenateChain import ConcatenateChain
from modules.llm_makersuite import MakerSuite

llm = MakerSuite(max_output_tokens=800)

from tools.DiceRoller import DiceRoller
from tools.dndSRD import dndSRD

# Set the memory to go back 4 turns
window_memory = ConversationBufferWindowMemory(k=4)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
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
    tools=[dndSRD(), DiceRoller()],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

full_chain = ConcatenateChain(agent=agent, conversation=conversation)

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
