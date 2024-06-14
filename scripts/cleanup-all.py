import os
from pinecone import Pinecone


def main():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", None))

    for collection in pc.list_collections().names():
        try:
            print("Deleting collection: " + collection)
            pc.delete_collection(collection)
        except Exception as e:
            print("Failed to delete collection: " + collection + " " + str(e))
            pass

    for index in pc.list_indexes().names():
        try:
            print("Deleting index: " + index)
            pc.delete_index(index)
        except Exception as e:
            print("Failed to delete index: " + index + " " + str(e))
            pass


if __name__ == "__main__":
    main()
