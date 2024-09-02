import time


def test_ollama_embedding():
    from llama_index.embeddings.ollama import OllamaEmbedding
    ollama_embedding = OllamaEmbedding(model_name="mxbai-embed-large")
    print("ollama_embedding_len= ", len(ollama_embedding.get_text_embedding("Hello world")))


def test_ollama_vision():
    import ollama

    start_time = time.time()
    res = ollama.chat(
        model="moondream",
        messages=[
            {
                'role': 'user',
                'content': 'Question: describe what you see in this image. Answer:',
                'images': ['./2_0.png']
            }
        ]
    )
    elapsed_time = round(time.time() - start_time, 2)
    print("Elapsed time: {} seconds".format(elapsed_time))

    print(res['message']['content'])


if __name__ == '__main__':
    test_ollama_embedding()
    # test_ollama_vision()
