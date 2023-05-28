import time

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
    "list",
    "write [test.js] console.log(\"Hello world\")",
    "describe [test.js] just a Hello World script",
    """write [main.js] <<<
const arr = ["Hello", "World"]
console.log("Hello world")
console.log("Hello world again")
<<<""",
    "list",
]

for tool in toolbelt:
    print(f"==== {tool.name} ====")
    print("================")
    print(tool.description)
    print("================")
    for query in input_queries:
        start_timer = time.time()
        result = tool._run(query)
        print(f"[query] {query}")
        print(f"[timer] ({time.time() - start_timer})")
        print(f"[result] {result}")
        