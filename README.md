# devcon2024

#the implementation process 
--------

1. Load documents from folder
2. Split documents
3. Create vector embeddings
4. Store vector embeddings
5. Get user query
6. Create vector embeddings of question
7. Conduct dense retrieval using a semantic search of vector database]
8. Filter the top 3 chunks of data according to similarity score
9. Generate promt and combine user query, context and prompt config
10. Send prompt to LLM
11. LLM will generate an accurate and reliable answer