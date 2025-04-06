import pandas as pd
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path="app/resources/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        # Ensure skills is a list of strings
        if isinstance(skills, dict):
            skills = list(skills.values())  # Convert dictionary values to list

        if not isinstance(skills, list):
            skills = [str(skills)]  # Convert single skill to list

        results = self.collection.query(query_texts=skills, n_results=2)

        if results and 'metadatas' in results and results['metadatas']:
            return [meta["links"] for meta in results['metadatas'][0] if "links" in meta]
        return []

