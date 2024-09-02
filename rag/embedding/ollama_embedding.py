def get_ollama_embedding_model():
    from llama_index.embeddings.ollama import OllamaEmbedding
    ollama_embedding = OllamaEmbedding(
        model_name="mxbai-embed-large",
        ollama_additional_kwargs={"mirostat": 0},
    )
    return ollama_embedding


def llamaindex_test():
    ollama_embedding = get_ollama_embedding_model()
    query_embedding = ollama_embedding.get_query_embedding("Where is blue?")
    print(query_embedding)


def origin_test():
    import ollama
    response = ollama.embeddings(
        model='mxbai-embed-large:latest',
        prompt='Llamas are members of the camelid family',
    )
    print(len(response['embedding']))


if __name__ == '__main__':
    # origin()
    llamaindex_test()
