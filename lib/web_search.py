from duckduckgo_search import DDGS
import time
import random


def web_search(query, max_results=3, max_retries=3):
    """
    Perform a web search with retry logic for rate limiting.

    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        max_retries (int): Maximum number of retry attempts

    Returns:
        list: List of search results
    """
    retry_count = 0
    base_delay = 1  # Base delay in seconds

    while retry_count < max_retries:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return []

                formatted_results = []
                for result in results:
                    formatted_results.append(
                        {
                            "title": result["title"],
                            "link": result["href"],
                            "body": result["body"],
                        }
                    )

                return formatted_results

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Ratelimit" in error_str:
                retry_count += 1
                if retry_count < max_retries:
                    # Calculate exponential backoff with jitter
                    delay = (base_delay * 2**retry_count) + (random.random() * 0.1)
                    print(f"Rate limited, retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue

            print(f"Error performing web search: {error_str}")
            return []

    print("Max retries reached for web search")
    return []
