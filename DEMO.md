# Overview / Demo
Note: I'm by no means a python expert, this is also a slapped together side  project

1. [Main](./main.py) imports and runs LangChain
    - Two chains (`conversationalist`, and `agent`)
2. The [Tool](./tools/Wikipedia.py)s are called by `agent` because of LangChain's [MRKL](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/mrkl/prompt.py) prompt
    - The tool class's `description` tells the `agent` when and how to talk with the tool
3. [ConcatenateChain](./modules/chain_concatenate.py) links both chains by having the `agent`'s text appear as a 3rd person in the chat prompt template called "Helper:"
4. Demo: `python .\main.py`
   - Please create a Gnome Wizard for my D&D campaign. Please make up a name and backstory for this character who lives in Waterdeep

