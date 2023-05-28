import time
from colorama import Fore, Style

from langchain.agents import initialize_agent, AgentType
from langchain.tools import HumanInputRun

from ide_tools.FileManager import FileManager
from modules.llm_makersuite import MakerSuite
llm = MakerSuite()

agent = initialize_agent(
    tools=[
        FileManager(),
        HumanInputRun(input_func=input),
    ],
    llm = llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

initial_prompt = """Think about your problem step-by-step to find a solution.
You are an single software engineer working remotely with your peers on a coding project.
The overall goal of the program is to count from 1 to 10
This program is written in node (JavaScript).

Please list, write, describe, and run files as needed to accomplish your goal

Perform the following task today in this order:
1) Start the project
2) Write an constant.js file which contains and exports a const variable `numberList` array
3) Write a helper.js file that will import and iterate over the numberList variable from step 2, running a console.log() for each number in the array
4) Write a main.js file that imports the function file and calls the function
"""

agent.run(initial_prompt)