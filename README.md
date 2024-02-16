# Generative-AI-QA-Model
In this project, on receiving a query from the user, it finds the relevant extracts from the book and generates answer based on the query and the extracts
# File desciption
web_extractiion.py- Extracts dtaa from the website by using **MarkupLm**
rag.py- Create sentence embedding of the text files and updates vector database (Chroma)
deploy/app.py- Deploy LLM model (Mistral-7M) by using Fast API. this Api accepts query as an input, finds relevent text extract from vector database by using MMR and then combine those extract with prompt and generates answer

# Output
![image](https://github.com/Alekh-sinha/Generative-AI-QA-Model/assets/22698201/0d37ca0b-3b8b-4dcd-b0ff-e5879c846e98)
In this image, blue box indicates context or relevant text extracted from book (relevancy is based on the question). Lower box is the answer which LLM model developed based on the context  
