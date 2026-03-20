Here, we will implement RAG to parse user input as query our backline line program with natural text. 

I will lay out the suggested task:

1) Turn the CSV into Documents using the Python f-string template. Add metadata to every document on ingest.

2) Use text-embedding to put them into a vector store. Chroma is advised so we will be able to do more complex operations with the metadata.

3) When a user asks a question, retrieve only the most relevant matches and use. Mostly the matches involved will be the primary teams' previous matches or the opposing teams' previous matches. It could be less than this depending on the query from the user.

4) Send those matches to the LLM to Compare with this game and provide the user some thoughts, hit rates and tactical trends. We want to avoid making predictions.