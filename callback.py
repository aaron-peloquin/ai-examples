# from colorama import Fore, Style
import time
start_time = time.time()

from langchain.chains import LLMChain
from langchain import PromptTemplate

from modules.callback_custom import CustomCallbackHandler
from modules.llm_makersuite import MakerSuite

from tools.DiceRoller import DiceRoller
from tools.dndSRD import dndSRD
from tools.DND5E import DND5E

llm = MakerSuite()
llm_time = time.time()
print(f"==llm'ed== {llm_time - start_time}")

handler = CustomCallbackHandler()

template = """Robot is a chatbot named "Cap'n Arrrrr" who always answers questions like a pirate

Human: {human_text}
Robot: """

prompt = PromptTemplate(
    input_variables=["human_text"],
    template=template,
)

lang_chained_chain_for_chaining = LLMChain(
    callbacks=[handler],
    llm=llm,
    prompt=prompt
)
chain_time = time.time()
print(f"==chained== {chain_time - llm_time}")

reply = lang_chained_chain_for_chaining.run(human_text="What is your most prized possession?")

# The LLM named itself
print(f"Pirate Captain Arrrrr: {reply}")
