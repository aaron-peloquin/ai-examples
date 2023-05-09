import random
import re
from typing import Optional

from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class DiceRoller(BaseTool):
    """Tool that adds the capability to roll dice like 1d6+3."""

    name = "DICEROLLER"
    description = (
        # "A tool who is an expert at rolling dice "
        "Used for when you need to roll dice to get a random number "
        "The Action Input should be a single string in the style of tabletop gaming, examples: `1d20`, `3d6`, `2d4`, etc." # 2d4+2
        # " Dice Roller does not know anything about characters or the world. Do not cite any skill names or ability scores as strings when using this tool. "
        # "Instead look up the number that represents that and send the actual value (number) to DiceRoller"
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the DiceRoller tool."""
        print("")
        print(f"= DiceRoller tool query is `{query}`")
        match = re.search(r"\d+[Dd]\d+", query)
        if match:
            diceString = match.group(0)
        else:
            match = re.search(r"[Dd]\d+", query)
            if match:
                diceString = f"1{match.group(0)}"
            else:
                return f"Invalid Agent Input syntax ({query}), try again with syntax like `#d#` where # is a number"

        print(f"= diceString is {diceString} =")

        num_dice, sides = diceString.lower().split("d")
        num_dice = int(num_dice)
        sides = int(sides)
        print(f"= parsed as {num_dice} D {sides}= ")
        outcome = self.roll_dice(num_dice, sides)
        print(f"= rolled {outcome}")
        return f"Rolled {diceString}, total result: {outcome}"

    def roll_dice(self, num_dice, sides):
        """Return a list of integers with length `num_dice`.

        Each integer in the returned list is a random number between 1 and `sides`, inclusive.
        """
        roll_results = 0
        for _ in range(num_dice):
            roll = random.randint(1, sides)
            roll_results += roll
            print(f"= agent rolled 1d{sides}... got {roll}, totaling {roll_results} so far =")
        return roll_results

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the DiceRoller tool asynchronously."""
        raise NotImplementedError("DiceRoller does not support async")
