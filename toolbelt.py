import time
from tools.DND5E import DND5E
# from tools.DiceRoller import DiceRoller
# from tools.EquationSolver import EquationSolver
# from tools.dndSRD import dndSRD


toolbelt = [
    # EquationSolver(),
    # DiceRoller(),
    DND5E(),
    # dndSRD(),
]

input_queries = ["armor/leather armor", "weapon/maul", "magic item/potion+of healing"]

for tool in toolbelt:
    print(f"==== {tool.name} ====")
    print("================")
    print(tool.description)
    print("================")
    for query in input_queries:
        print(f"[query] {query}")
        start_timer = time.time()
        result = tool._run(query)
        print(f"[timer] ({time.time() - start_timer})")
        print(f"[result] {result}")
        