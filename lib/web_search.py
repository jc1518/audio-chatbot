from duckduckgo_search import DDGS


def web_search(query, max_results=3):
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return []

        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "title": result["title"],
                    "link": result["link"],
                    "body": result["body"],
                }
            )

        return formatted_results
    except Exception as e:
        print(f"Error performing web search: {str(e)}")
        return []
