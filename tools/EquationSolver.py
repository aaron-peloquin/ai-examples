from typing import Optional

import numexpr
import re
import operator as op

from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

class EquationSolver(BaseTool):
    """Tool that adds the capability to solve equations like 2 + 2."""

    name = "EQUATIONSOLVER"
    description = (
        # "A tool who is an expert at solving equations "
        "Used for when you need to solve an equation to get a result "
        "The Action Input should be a single string in the style of mathematics, examples: `2 + 2`, `3 * 4`, `2 - 1`, `2 / 1`, etc. " # 2d4+2
        "Does not perform math on variables, so `x + 3` is not acceptable input"
        # " Equation Solver does not know anything about characters or the world. Do not cite any skill names or ability scores as strings when using this tool. "
        # "Instead look up the number that represents that and send the actual value (number) to Equation Solver"
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the EquationSolver tool."""
        print("")
        print(f"==== EquationSolver qry: `{query}`")

        # Strip all characters that do not match the regular expression.
        equation_string = query.replace("+", " + ").replace("-", " - ").replace("*", " * ").replace("/", " / ")
        equation_tokens = equation_string.split()
        equation_parsed = " ".join(equation_tokens)



        solution = numexpr.evaluate(equation_parsed)
        print(f"== Solution == ({equation_parsed}) = ({solution})")
        return solution

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the EquationSolver tool asynchronously."""
        raise NotImplementedError("EquationSolver does not support async")
