import os
import re
from pinecone import Pinecone
from datetime import datetime, timedelta


def delete_everything(pc):
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


def parse_date(resource_name):
    match = re.search(r"-\d{8}-", resource_name)
    if match:
        date_string = match.group(0).strip("-")
        return datetime.strptime(date_string, "%Y%m%d")
    else:
        return None


def is_resource_old(resource_name):
    print(f"Checking resource name: {resource_name}")
    resource_datetime = parse_date(resource_name)
    if resource_datetime is None:
        return False
    current_time = datetime.now()

    # Calculate the difference
    time_difference = current_time - resource_datetime

    # Check if the time difference is greater than 24 hours
    print(f"Resource timestamp: {resource_datetime}")
    print(f"Time difference: {time_difference}")
    return time_difference > timedelta(hours=24)


def delete_old(pc):
    for collection in pc.list_collections().names():
        if is_resource_old(collection):
            try:
                print("Deleting collection: " + collection)
                pc.delete_collection(collection)
            except Exception as e:
                print("Failed to delete collection: " + collection + " " + str(e))
                pass
        else:
            print("Skipping collection, not old enough: " + collection)

    for index in pc.list_indexes().names():
        if is_resource_old(index):
            try:
                print("Deleting index: " + index)
                pc.delete_index(index)
            except Exception as e:
                print("Failed to delete index: " + index + " " + str(e))
                pass
        else:
            print("Skipping index, not old enough: " + index)


def main():
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", None))

    if os.environ.get("DELETE_ALL", None) == "true":
        print("Deleting everything")
        delete_everything(pc)
    else:
        print("Deleting old resources")
        delete_old(pc)


if __name__ == "__main__":
    main()
