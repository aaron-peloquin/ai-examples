# Overview / Demo
Note: I am not a python expert, this is a rough side project used for learning

1. [Main](./main.py) imports and runs LangChain. Creates 2 chains (`conversationalist`, and `agent`)
2. Tools like [Wikipedia](./tools/Wikipedia.py) are called by `agent` because of LangChain's [MRKL](https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/mrkl/prompt.py) system (Modular Reasoning, Knowledge, and Language)
    - The tool class's `description` tells the `agent` when and how to talk with the tool
3. [ConcatenateChain](./modules/chain_concatenate.py) links both chains by having the `agent`'s text appear as a 3rd person in the chat prompt template called "Helper:"
4. Demo: `python .\main.py`
   - Please create a lawful good Half-Orc Wizard for my D&D campaign. Please make up a name and backstory for this character who lives somewhere in Waterdeep

