import time
from colorama import Fore, Style

from langchain.agents import initialize_agent, AgentType
from langchain.chains import SimpleSequentialChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from modules.llm_vicuna7b import llm
from tools.DiceRoller import DiceRoller
from tools.srdReader import srdReader

# Set the memory to go back 4 turns
window_memory = ConversationBufferWindowMemory(k=4)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=window_memory,
)

conversation.prompt.template = '''The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context.

If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
----

Human: {input}
(Additional Information: {context})
AI:'''

# AI cannot roll dice on its own and should not make up random numbers, instead AI will use the "DiceRoller" Action to roll dice.
# When AI is unsure of the content and rules of Dungeons and Dragons (D&D), AI will use the "DNDSRD" Action to retrieve excerpts from the D&D rules book.

agent = initialize_agent(
    tools=[srdReader(), DiceRoller()],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

dndQuestion = "Please roll 3d6 to determine my Charisma ability score"
print(Fore.BLUE, f"Humanoid: {dndQuestion}", Style.RESET_ALL)
agent_reply = agent.run(dndQuestion)
print(Fore.LIGHTMAGENTA_EX, f"D&D Bot: {agent_reply}")

dndQuestion = "(In Dungeons and Dragons) What age are Halflings considered mature?"
print(Fore.BLUE, f"Humanoid: {dndQuestion}", Style.RESET_ALL)
agent_reply = agent.run(dndQuestion)
print(Fore.LIGHTMAGENTA_EX, f"D&D Bot: {agent_reply}")

# full_chain = SimpleSequentialChain(
#     chains=[agent, conversation],
#     input_key="context",
#     output_key="context",
#     verbose=True,
# )

# print(Fore.RED, '==== STARTING ====')
# print(Style.RESET_ALL)
# starter_prompt = "(In Dungeons and Dragons) What age are Halflings considered mature?"
# print(f"== starter_prompt ({starter_prompt}) ==")
# print(full_chain.run(starter_prompt))

# while True:
#     print(Fore.CYAN)
#     human_input = input("Human:")
#     start_time = time.time()
#     print(Style.RESET_ALL)
    
#     # ai_chat_reply = conversation.run(input=human_input)
#     # ai_agent_reply = agent.run(human_input)
#     ai_chat_agent_reply = full_chain.run(human_input)

#     print(Fore.LIGHTGREEN_EX, ai_chat_agent_reply, Style.RESET_ALL)
    
#     print(f"== {time.time() - start_time} ==")
