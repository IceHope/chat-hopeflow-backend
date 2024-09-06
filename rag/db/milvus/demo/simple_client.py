from typing import List

from pymilvus import DataType
from pymilvus import FieldSchema, CollectionSchema, MilvusClient

from rag.managers.chunk_manager import ChunkManager
from rag.managers.embedding_manager import EmbeddingManager
from rag.managers.reader_manager import ReaderManager

reader_manager = ReaderManager()

chunk_manager = ChunkManager()

embedding_manager = EmbeddingManager("bge-small-en-v1.5")

COLLECTION_NAME = "simple_client"


class FieldName:
    ID = "id"
    EMBEDDING = "embedding"
    CHUNK = "chunk"


def _init_client():
    client = MilvusClient(
        uri="http://localhost:19530",
    )
    return client


def _init_schema():
    # 1 schema_field
    field_schemas = [
        FieldSchema(name=FieldName.ID, dtype=DataType.INT64, auto_id=False, is_primary=True),
        FieldSchema(name=FieldName.EMBEDDING, dtype=DataType.FLOAT_VECTOR, dim=embedding_manager.get_dim(),
                    description="vector"),
        FieldSchema(name=FieldName.CHUNK, dtype=DataType.VARCHAR, max_length=5000, description="chunk docs")
    ]

    # 2 schema_collection
    return CollectionSchema(
        enable_dynamic_field=True,
        fields=field_schemas,
        description="desc of a collection")


def _init_index_params():
    index = MilvusClient.prepare_index_params()

    index.add_index(
        field_name=FieldName.EMBEDDING,
        index_type="IVF_FLAT",
        metric_type="COSINE",
        index_name="vector_index",
        params={"nlist": 128}
    )

    index.add_index(
        field_name=FieldName.CHUNK,
        index_type="",  # Type of index to be created. For auto indexing, leave it empty or omit this parameter.
        index_name="default_index"  # Name of the index to be created
    )

    return index


def get_embeddings_from_chunk(chunks: List[str]):
    return embedding_manager.get_model().get_text_embedding_batch(texts=chunks, show_progress=True)


def _init_data():
    paths = ["./data/paul_graham_essay.txt"]
    documents = reader_manager.load_file_list(input_file_paths=paths)

    nodes = chunk_manager.chunk_documents(documents=documents)

    chunks = [node.text for node in nodes]

    embeddings = get_embeddings_from_chunk(chunks=chunks)

    return [
        {FieldName.ID: i, FieldName.EMBEDDING: embeddings[i], FieldName.CHUNK: chunks[i]}
        for i in range(len(chunks))
    ]


def create_partition(client, partition_name: str, collection_name: str):
    res = client.list_partitions(collection_name=collection_name)
    print(res)

    client.create_partition(
        collection_name=collection_name,
        partition_name=partition_name
    )

    res = client.list_partitions(collection_name=collection_name)
    print(res)


def _load_datas():
    # 1. client
    milvus_client = _init_client()

    # 2. init schema
    collection_schema = _init_schema()

    # 3. init index params
    index_params = _init_index_params()

    # 4. init data
    data = _init_data()

    # 5. drop collection if exists
    if milvus_client.has_collection(collection_name=COLLECTION_NAME):
        milvus_client.drop_collection(collection_name=COLLECTION_NAME)

    # 6. create collection
    milvus_client.create_collection(
        enable_dynamic_field=True,
        collection_name=COLLECTION_NAME,
        schema=collection_schema,
        index_params=index_params,
        dimension=embedding_manager.get_dim()
    )

    partition_name = "partitionA"
    # 6.1 create partition
    create_partition(client=milvus_client, partition_name=partition_name, collection_name=COLLECTION_NAME)
    # 7. insert data
    res = milvus_client.insert(collection_name=COLLECTION_NAME, data=data, partition_name=partition_name)
    print(res)


def _search_data(client: MilvusClient, queries: List[str]):
    query_vectors = get_embeddings_from_chunk(queries)

    res = client.search(
        collection_name=COLLECTION_NAME,  # target collection
        data=query_vectors,  # query vectors
        limit=2,  # number of returned entities
        output_fields=[FieldName.CHUNK],  # specifies fields to be returned
        partition_names=["partitionB"]
    )
    print(res)
    # for q in queries:
    #     print("Query:", q)
    #     for result in res:
    #         print_json(result)
    #     print("\n")


if __name__ == "__main__":
    # _load_datas()

    milvus_client = _init_client()
    # create_partition(client=milvus_client, partition_name="partitionB", collection_name=COLLECTION_NAME)

    # queries = ["人工智能的政策", "llama2有多少种参数"]
    queries = ["What I Worked On February 2021"]
    _search_data(queries=queries, client=milvus_client)
