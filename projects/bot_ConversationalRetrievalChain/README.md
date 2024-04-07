api_key = "your_api_key_here"
bot = ConversationalRetrievalChain(api_key)
query = "What is the weather today?"
response = bot.get_conversational_response(query)
print(response)
