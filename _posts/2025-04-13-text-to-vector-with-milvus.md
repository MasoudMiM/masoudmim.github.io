---
layout: post
title:  "Turning Text into a Vector Database with Milvus"
date: 2025-04-13 22:15:00
description: Simple tutorial for converting a text to a vector database.
tags: vector database, Milvus
categories: technical
---

In this tutorial, I am going to provide the steps required to convert a text into a vector database for easy searching and retrieval. The goal is to walk you through the process using a Python script that uses [Milvus](https://milvus.io/) and [Sentence Transformers](https://huggingface.co/sentence-transformers). You can find the complete script in this [GitHub repo](https://github.com/MasoudMiM/text-to-vector).

## What You Need

Before we get started, make sure you have the following:

1. **Milvus**: You need a running instance of Milvus. Check out the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) if you haven't set it up yet.
2. **Python Packages**: Install the required packages with this command:

   ```bash
   pip install nltk sentence-transformers numpy pymilvus
   ```

3. **Your Text**: Have a text file ready that you want to convert into a vector database. Just name it `text.txt` and place it in a `data` folder.

## The Code

The script reads your text, processes it, and stores it in a Milvus collection. Here's a quick overview of how it works:

1. **Read the Text**: The script starts by reading the content of your `text.txt` file.
    ```python
    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    ```
2. **Tokenization**: It breaks the text into sentences and then chunks them into manageable pieces.
    ```python
    sentences = sent_tokenize(text)
    ```
3. **Generate Embeddings**: Using the `all-MiniLM-L6-v2` model from Sentence Transformers, the script converts these text chunks into numerical embeddings.
    ```python
    model = SentenceTransformer('all-MiniLM-L6-v2')

    text_chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    embeddings = model.encode(text_chunks)
    ```
4. **Connect to Milvus**: It establishes a connection to your Milvus instance.
    ```python
    def connect_to_milvus():
        connections.connect("default", host="localhost", port="19530")
    ```
5. **Create a Collection**: A new collection is created in Milvus to store the embeddings and the original text.
    ```python
    connect_to_milvus()
    collection_name = MILVUS_COLLEC_NAME

    def create_milvus_collection(collection_name):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)  
        ]
        schema = CollectionSchema(fields, description="Vector database for Common Sense.")
        collection = Collection(name=collection_name, schema=schema)
        return collection

    collection = create_milvus_collection(collection_name)
    ```
6. **Insert Data**: Finally, the embeddings and text are inserted into the collection, making them ready for efficient searching.
    ```python
    def insert_vectors_to_milvus(collection, embeddings_list, original_texts):
        logging.info(f"Number of embeddings: {len(embeddings_list)}")
        
        data = [
            {"embedding": embedding, "text": original_text} 
            for embedding, original_text in zip(embeddings_list, original_texts)
        ]
        
        try:
            collection.insert(data)  
            logging.info(f"Inserted {len(embeddings_list)} vectors into Milvus.")
        except Exception as e:
            logging.error(f"Failed to insert data into Milvus: {e}")

    insert_vectors_to_milvus(collection, embeddings_list, text_chunks)
    ```

Once you have everything set up, just run the script:

```bash
python vdb_gen.py
```

Make sure to replace `vdb_gen.py` with the actual name of your Python file. The script will handle the rest!

## What's Next?

After running the script, you'll have a Milvus collection filled with your text data, ready for querying. You can now perform similarity searches and retrieve relevant information quickly and efficiently. Here is an example of how the output collection can look like in [attu interface](https://milvus.io/docs/v2.0.x/attu.md).

<div style="text-align: center;">
    <img src="/assets/img/db_example_attu.png" alt="attu showing the vector database" width="600">
</div>

You have now successfully converted text into a vector database using Milvus. One exciting application of this approach is in Retrieval-Augmented Generation (RAG). By storing your text data in a vector database, you can enhance the capabilities of language models. When generating responses, the model can retrieve relevant information from the database, leading to more accurate and contextually relevant outputs. This is particularly useful in applications like chatbots, content generation, and question-answering systems.
