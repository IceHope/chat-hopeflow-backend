COLLECTION_NAME_PREFIX = "hope_test_"


def get_db_collection_name(embedding_simple_name: str):
    return COLLECTION_NAME_PREFIX + embedding_simple_name
