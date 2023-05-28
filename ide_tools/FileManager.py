import os
import re
from typing import Optional

from langchain.tools.base import BaseTool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun

code_dir = ".\\ai-written-code\\"

class FileManager(BaseTool):
    """Tool that adds the capability to manage a filesystem."""

    name = "FileManager"
    description = (
"""Used to view and modify files within a code projects working directory.

Always wrap filenames in square brackets `[example.txt]`.
Each file should be a complete script file.
You have no control over making or entering other directories

The Action Input should be a single command to view or manage the project's files:

- `list` will return a list of filenames along with their descriptions.
- `describe [test.php] /**
 * @var is_caps function
 * Test for uppercase.
 * 
 * @param string $my_word text to be tested
 * @return bool false if not uppercase
 */` will set the description of that file in DocBlock format, descriptions are shown in `list`. Ensure you make one DocBlock for each the file's exported variables
- `run [filename.py]` will execute a single code script and return the terminal output to you.
- `open [example.ts]` Will open the file and return its current contents to you.
- `write [test.js] <<<export const test = ["foo", "bar"]
export const five = 5
<<<` Only works immediately following the `open` command.
Will completely overwrite the file's contents with whatever code is between the wrapping three less than symbols <<<`

To simplify your work, we do not support modifying existing files, making directories, or changing directories.
"""
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        print("")
        print(f"==== FileSystem qry: `{query}`")

        fsCmd, fsFilename, fsExtra = self.parseCmd(query)

        if fsCmd == "list":
            files = []
            file_number = 0
            for file in os.listdir(code_dir):
                if file.endswith(".describe"):
                    continue
                file_number += 1
                description = "(No Description metadata set)"
                try:
                    desc_file = f"{code_dir}{file}.describe"
                    # print(f"opening desc file: ({desc_file})")
                    with open(desc_file, "r") as f:
                        description = f.read()
                except FileNotFoundError:
                    pass
                size = os.path.getsize(os.path.join(code_dir, file))
                files.append((file, size, description, file_number))
            return "\n ".join([f"[file #{file_number}] {file} ({size} Bytes): {description}" for file, size, description, file_number in files])
        # elif fsCmd == "make":
        #     try:
        #         open(f"{code_dir}{fsFilename}", "w")
        #         # os.remove(f"{code_dir}{fsFilename}.describe")
        #         return f"created file {fsFilename}"
        #     except:
        #         return f"failed to create file {fsFilename}"

        elif fsCmd == "describe":
            try:
                with open(f"{code_dir}{fsFilename}.describe", "w") as f:
                    f.write(fsExtra)
                    return f"set description for {fsFilename}"
            except:
                return f"failed to set description for {fsFilename}"

        elif fsCmd == "run":
            print("running? lets just not yet ok...")
            # os.system(f"node {fsFilename}")
        elif fsCmd == "open":
            try:
                with open(f"{code_dir}{fsFilename}", "r") as f:
                    contents = f.read()
                    return f"opened {fsFilename}, its contents are:\n\n{contents}"
            except:
                return f"failed to open {fsFilename}"

        elif fsCmd == "write":
            try:
                with open(f"{code_dir}{fsFilename}", "w") as f:
                    f.write(fsExtra)
                    print(f"writing file contents ({fsExtra})")
                    return f"wrote to file {fsFilename}"
            except:
                return f"failed to write to file {fsFilename}"
        else:
            return f"Unknown command: {fsCmd}"

    def parseCmd(self, query):
        fsCmd = "error"
        fsFilename = ""
        fsExtra = ""

        match = re.match(r'^(?P<command>[a-zA-Z]+)\s*(?:(?P<filename>\[[a-zA-Z0-9._-]+\]))?\s*(?P<text>.*)$', query, flags=re.DOTALL)

        if match:
            fsCmd = match.group("command")
            try:
                fsFilename = match.group("filename")
                fsFilename = fsFilename.replace("[", "").replace("]", "")
            except:
                fsFilename = ""
            try:
                fsExtra = match.group("text") or ""
                fsExtra = fsExtra.replace("<<<", "")
            except:
                fsExtra = ""

            fsCmd = fsCmd.strip()
            fsFilename = fsFilename.strip()
            fsExtra = fsExtra.strip()

        # print(f"fsCmd:{fsCmd}")
        # print(f"fsFilename:{fsFilename}")
        # print(f"fsExtra:{fsExtra}")

        return fsCmd, fsFilename, fsExtra

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the FileManager tool asynchronously."""
        raise NotImplementedError("FileManager does not support async")
