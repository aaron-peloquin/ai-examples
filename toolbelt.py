import time
from colorama import Fore, Style

from ide_tools.FileManager import FileManager
# from tools.DND5E import DND5E
# from tools.DiceRoller import DiceRoller
# from tools.EquationSolver import EquationSolver
# from tools.dndSRD import dndSRD


toolbelt = [
    # EquationSolver(),
    # DiceRoller(),
    # DND5E(),
    # dndSRD(),
    FileManager()
]

input_queries = [
#    "list",
#     "write [test.js] console.log(\"Hello world\")",
#     "describe [test.js] just a Hello World script",
#     """write [main.js] <<<
# const arr = ["Hello", "World"]
# console.log("Hello world")
# console.log("Hello world again")
# <<<""",
    "list",
    "open [main.js]",
]

for tool in toolbelt:
    print(f"==== {tool.name} ====")
    print("================")
    print(tool.description)
    print("================")
    for query in input_queries:
        start_timer = time.time()
        result = tool._run(query)
        duration = str(time.time() - start_timer)[0:8]

        print(f"[query] {Fore.CYAN}{query}{Style.RESET_ALL}")
        print(f"[timer] {duration}s")
        print(f"[result] {Fore.GREEN}{result}{Style.RESET_ALL}")
        