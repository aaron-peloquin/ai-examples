from typing import Optional
from langchain.tools.base import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
import wikipedia

class Wikipedia(BaseTool):
    """Tool that adds the capability to get information from Wikipedia"""

    name = "Wikipedia"
    description = (
        "Used to retrieve general information about anything from the real world. "
        "The search query should be simplified term like a single item or subject. If needed, add a coma after the term and add any extra details after the coma. "
        "This tool will return hard facts that you can trust, do not embellish or expand on information returned from this tool. "
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the DND5E tool."""
        print("")
        print(f"==== Wikipedia qry: `{query}`")
        query = str.lower(query)
        results = """Top Search Results:
        """

        search_results = wikipedia.search(query)

        if len(search_results) > 1:
            search_results = search_results[:1]
        for page in search_results:
            page_results = wikipedia.page(page)
            
            results += f"""{page_results.title}
            {page_results.content}"""

        return results
