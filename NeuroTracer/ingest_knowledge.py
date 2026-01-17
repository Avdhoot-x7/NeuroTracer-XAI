import chromadb
from chromadb.utils import embedding_functions

# 1. Initialize the Persistent Client (stores data in a folder)
client = chromadb.PersistentClient(path="./knowledge_base")

# 2. Use a standard embedding function (Sentence Transformers)
# This turns "Paris is in France" into a 384-dimensional vector
default_ef = embedding_functions.DefaultEmbeddingFunction()

# 3. Create a 'Collection' (Think of it as a Table in SQL)
collection = client.get_or_create_collection(name="facts", embedding_function=default_ef)

# 4. Add some 'Ground Truth' facts
# In a real project, you'd load these from a PDF or Wikipedia
facts = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Japan is Tokyo.",
    "Mars is a planet and has no human-built capitals yet.",
    "The Earth revolves around the Sun."
]

ids = [f"id{i}" for i in range(len(facts))]

collection.add(
    documents=facts,
    ids=ids
)

print(f"✅ Indexed {len(facts)} facts into the Vector Database.")