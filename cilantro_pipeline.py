import chromadb
from pypdf import PdfReader
from embedding_helper import create_embedding


# Pipeline configuration and helpers
PDF_PATH = "./static/200_Comprehensive_Recipes.pdf"
DB_PATH = "./final_project_db"
COLLECTION_NAME = "recipe_knowledge_ollama"


def recipe_title(recipe: str) -> str:
    lines = [line.strip() for line in recipe.splitlines() if line.strip()]
    return lines[1] if len(lines) > 1 else "Untitled recipe"


# Load and embed recipe text
reader = PdfReader(PDF_PATH)
recipes = []
for page in reader.pages:
    page_text = page.extract_text() or ""
    if page_text.startswith("RECIPE"):
        recipes.append(page_text.strip())

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

print("Creating Ollama embeddings...")
embeddings = [create_embedding(recipe) for recipe in recipes]

# Store recipe chunks and metadata in Chroma
ids = []
metadatas = []
for index, recipe in enumerate(recipes, start=1):
    number = f"{index:03d}"
    ids.append(f"recipe_{number}")
    metadatas.append(
        {
            "recipe_number": number,
            "title": recipe_title(recipe),
            "source": "200_Comprehensive_Recipes.pdf",
        }
    )

collection.upsert(
    documents=recipes,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids,
)
print(f"Indexed recipes: {len(recipes)}")
print(f"Collection count: {collection.count()}")
print(f"Embedding dimensions: {len(embeddings[0])}")
