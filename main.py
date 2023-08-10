import time
from colorama import Fore, Style

from langchain.agents import initialize_agent, AgentType
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from modules.chain_concatenate import ConcatenateChain
# from modules.callback_custom import CustomCallbackHandler
from modules.llm_makersuite import MakerSuite
from tools.EquationSolver import EquationSolver
from tools.Wikipedia import Wikipedia
llm = MakerSuite()

from tools.DiceRoller import DiceRoller
from tools.dndSRD import dndSRD
from tools.DND5E import DND5E

# Set the memory to go back 12 turns
window_memory = ConversationBufferWindowMemory(k=12)

agent = initialize_agent(
    tools=[
        DND5E(),
        DiceRoller(),
        EquationSolver(),
        dndSRD(),
        Wikipedia()
    ],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

conversationalist = ConversationChain(
    llm=llm,
    verbose=False,
    memory=window_memory,
)

conversationalist.prompt.template = '''This is a chat log of an engaging conversation between an AI Assistant and a Human about Dungeons and Dragons (D&D).
The Assistant is a creative storyteller and will assist with creative ideas.
The Helper gathers information and takes actions such as rolling dice on behalf of the Assistant.
Assistant's job is to be the storyteller, not to know rules or perform calculations.
The AI Assistant should describe a location in detail, including the physical features, the atmosphere, and any notable inhabitants.
Assistant will trust that Helper did their job of retrieving truthful information completely and transcribe that same information to the human in a more engaging way without mentioning or giving thanks to Helper.
The Assistant does not embellish, change, or make up information that Helper retrieved.
If Assistant creates a puzzle that is challenging but fair. The puzzle should have a clear solution, but it should not be too easy to solve.


Current conversation:
{history}
----
Human: {input}
Assistant: '''

full_chain = ConcatenateChain(agent=agent, conversation=conversationalist)

print(Fore.RED, '====', Style.RESET_ALL, ' STARTING ', Fore.RED, '====')

while True:
    print(Fore.CYAN)
    human_input = input("Human: ")
    start_time = time.time()
    print(Style.RESET_ALL)

    try:
        ai_chat_agent_reply = full_chain.run(human_input)
    except ValueError as e:
        ai_chat_agent_reply = str(e)
        if not ai_chat_agent_reply.startswith("Could not parse LLM output: `"):
            raise e
        ai_chat_agent_reply = ai_chat_agent_reply.removeprefix("Could not parse LLM output: `").removesuffix("`")

    print(Fore.LIGHTMAGENTA_EX, ai_chat_agent_reply, Style.RESET_ALL)

    print(f"== {time.time() - start_time} ==")
