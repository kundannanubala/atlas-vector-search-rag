import os
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.docstore.document import Document
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
import key_param
import traceback

# Set up Google API
genai.configure(api_key=key_param.google_api_key)

# Custom Embedding class
class CustomEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Embedding of text"
        )
        return result['embedding']

# Set the MongoDB URI, DB, Collection Names
try:
    client = MongoClient(key_param.MONGO_URI)
    dbName = "langchain_demo"
    collectionName = "collection_of_text_blobs"
    collection = client[dbName][collectionName]
    print("Successfully connected to MongoDB")

    # Clear the collection
    collection.delete_many({})
    print("Cleared existing documents from the collection")
except Exception as e:
    print(f"Error connecting to MongoDB or clearing collection: {e}")
    exit(1)

# Read texts from files
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

sample_files_dir = './sample_files'
story_files = ['story1.txt', 'story2.txt', 'story3.txt']

texts = []
for file_name in story_files:
    file_path = os.path.join(sample_files_dir, file_name)
    if os.path.exists(file_path):
        content = read_text_file(file_path)
        texts.append((file_name, content))
    else:
        print(f"Warning: File {file_path} not found.")

# Create Document objects
data = [Document(page_content=content, metadata={"title": title}) for title, content in texts]

print(f"Created {len(data)} documents")

# Initialize the custom embedding
custom_embeddings = CustomEmbedding()

# Attempt to create MongoDBAtlasVectorSearch
print("Attempting to create MongoDBAtlasVectorSearch...")
try:
    vectorStore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=custom_embeddings,
        index_name="default"
    )
    print("Successfully created MongoDBAtlasVectorSearch")

    # Now let's try to add the documents
    print("Attempting to add documents to the vector store...")
    vectorStore.add_documents(data)
    print("Successfully added documents to the vector store")
except Exception as e:
    print(f"Error creating vector store or adding documents: {e}")
    print(traceback.format_exc())
    exit(1)

print("Script completed successfully")

# Test a simple similarity search
print("Testing similarity search...")
query = "A magical garden at night"
results = vectorStore.similarity_search(query)
print(f"Number of results returned: {len(results)}")
if results:
    print(f"Top result for '{query}': {results[0].page_content[:100]}...")
else:
    print("No results found. This could indicate an issue with the vector store or the search functionality.")

# Debug: Check the contents of the collection
print("\nDebug: Checking contents of MongoDB collection...")
docs_in_collection = list(collection.find({}))
print(f"Number of documents in collection: {len(docs_in_collection)}")
if docs_in_collection:
    print("Sample document:")
    print(docs_in_collection[0])
else:
    print("The collection is empty.")

# Additional debug: Check if documents have embeddings
print("\nChecking if documents have embeddings...")
docs_with_embeddings = list(collection.find({"embedding": {"$exists": True}}))
print(f"Number of documents with embeddings: {len(docs_with_embeddings)}")

# Check the index
print("\nChecking vector search index...")
indexes = collection.index_information()
print("Indexes:", indexes)