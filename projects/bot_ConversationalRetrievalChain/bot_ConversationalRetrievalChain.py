import requests


class ConversationalRetrievalChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://example.com/api"

    def process_query(self, query):
        payload = {"query": query, "api_key": self.api_key}
        response = requests.get(self.base_url, params=payload)
        return response.json()

    def retrieve_information(self, processed_query):
        info = self.process_query(processed_query["query"])
        return info

    def format_response(self, information):
        formatted_response = f"Here's what I found: {information['data']}"
        return formatted_response

    def get_conversational_response(self, query):
        processed_query = {"query": query}
        information = self.retrieve_information(processed_query)
        return self.format_response(information)

# Example usage
if __name__ == "__main__":
    api_key = "your_api_key_here"
    bot = ConversationalRetrievalChain(api_key)
    query = "What is the weather today?"
    response = bot.get_conversational_response(query)
    print(response)
