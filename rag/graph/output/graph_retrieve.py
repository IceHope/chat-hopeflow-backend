from langchain_community.vectorstores import Neo4jVector

NEO4J_URI = "bolt://localhost"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "AP585576ice"
NEO4J_DATABASE = "neo4j"  # 指定数据库名

topChunks = 3
topCommunities = 3
topOutsideRels = 10
topInsideRels = 10
topEntities = 10

lc_retrieval_query = """
WITH collect(node) as nodes
// Entity - Text Unit Mapping
WITH
collect {
    UNWIND nodes as n
    MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
    WITH c, count(distinct n) as freq
    RETURN c.text AS chunkText
    ORDER BY freq DESC
    LIMIT $topChunks
} AS text_mapping,
// Entity - Report Mapping
collect {
    UNWIND nodes as n
    MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
    WITH c, c.rank as rank, c.weight AS weight
    RETURN c.summary 
    ORDER BY rank, weight DESC
    LIMIT $topCommunities
} AS report_mapping,
// Outside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE NOT m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topOutsideRels
} as outsideRels,
// Inside Relationships 
collect {
    UNWIND nodes as n
    MATCH (n)-[r:RELATED]-(m) 
    WHERE m IN nodes
    RETURN r.description AS descriptionText
    ORDER BY r.rank, r.weight DESC 
    LIMIT $topInsideRels
} as insideRels,
// Entities description
collect {
    UNWIND nodes as n
    RETURN n.description AS descriptionText
} as entities
// We don't have covariates or claims here
RETURN {Chunks: text_mapping, Reports: report_mapping, 
       Relationships: outsideRels + insideRels, 
       Entities: entities} AS text, 1.0 AS score, {} AS metadata
"""
from langchain_community.embeddings import OllamaEmbeddings

index_name = "entity"

lc_vector = Neo4jVector.from_existing_index(
    OllamaEmbeddings(model="mxbai-embed-large"),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=index_name,
    retrieval_query=lc_retrieval_query,
)

docs = lc_vector.similarity_search(
    "What do you know about 孙悟空,",
    k=topEntities,
    params={
        "topChunks": topChunks,
        "topCommunities": topCommunities,
        "topOutsideRels": topOutsideRels,
        "topInsideRels": topInsideRels,
    },
)
print(docs[0].page_content)