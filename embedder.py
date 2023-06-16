from sentence_transformers import SentenceTransformer
import nltk
import json
import numpy as np
import tqdm

# Load the pre-trained SBERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')
nltk.download('punkt')


def encode_sentences(text):
    sentences = []
    embeddings = None

    for line in text.split("\n"):
        if line:
            _sentences = nltk.sent_tokenize(line)
            _sentences = [_s.strip() for _s in _sentences if len(_s) < 500]
            _embeddings = model.encode(_sentences)
            if _embeddings is None or _embeddings.ndim < 2:
                continue
            if embeddings is not None:
                sentences += _sentences
                embeddings = np.concatenate([embeddings, _embeddings], axis=0)
            else:
                sentences = _sentences
                embeddings = _embeddings

    return sentences, embeddings


def encode_and_save_sentences(text, filename):
    sentences, embeddings = encode_sentences(text)
    
    if embeddings is not None:
        data = {"sentences": sentences, "embeddings": embeddings.tolist()}
    else:
        data = {"sentences": [], "embeddings": []}
    
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)
    
    print(f"Embeddings stored in file: {filename}")
    return sentences, embeddings


def clean_message(text):
    """ Remove code blocks."""
    cleaned_lines = []
    num_quotes = 0
    for line in text.split("\n"):
        if line.startswith("```"):
            num_quotes += 1
            cleaned_lines.append("")
        if num_quotes % 2 == 0:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def check_and_remove_duplicates(sentences, embeddings):
    unique_embeddings = []
    unique_sentences = []
    
    for embedding, sentence in zip(embeddings, sentences):
        # Convert embedding to tuple so it can be hashed and compared
        embedding_tuple = tuple(embedding)
        
        if embedding_tuple not in unique_embeddings:
            unique_embeddings.append(embedding_tuple)
            unique_sentences.append(sentence)
    
    # Convert back to numpy arrays
    unique_embeddings = np.array([embedding for embedding in unique_embeddings])
    
    return unique_sentences, unique_embeddings


def encode_and_append(text, filename):
    with open(filename, "r") as file:
        data = json.load(file)
    sentences = data["sentences"]
    embeddings = np.array(data["embeddings"])

    text = clean_message(text)
    _sentences, _embeddings = encode_sentences(text)
    if _embeddings is None or _embeddings.ndim < 2:
        return
    if embeddings is not None and embeddings.ndim == 2:
        sentences += _sentences
        embeddings = np.concatenate([embeddings, _embeddings], axis=0)
    else:
        sentences = _sentences
        embeddings = _embeddings 

    sentences, embeddings = check_and_remove_duplicates(sentences, embeddings)

    data = {"sentences": sentences, "embeddings": embeddings.tolist()}
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def encode_and_save_conversation_from_file(input_file, output_file):
    with open(input_file, 'r') as file:
        conversation = json.load(file)

    sentences = []
    embeddings = None

    encoded_conversation = []
    for message in tqdm.tqdm(conversation["messages"]):
        content = message.get("content")
        if content:
            content = clean_message(content)
            _sentences, _embeddings = encode_sentences(content)
            if _embeddings is None or _embeddings.ndim < 2:
                continue
            if embeddings is not None:
                sentences += _sentences
                embeddings = np.concatenate([embeddings, _embeddings], axis=0)
            else:
                sentences = _sentences
                embeddings = _embeddings
    
    if embeddings is not None:
        data = {"sentences": sentences, "embeddings": embeddings.tolist()}
    else:
        data = {"sentences": [], "embeddings": []}

    with open(output_file, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Embeddings stored in file: {output_file}")
    return encoded_conversation


def retrieve_sentences(query, filename, top_n=5):
    with open(filename, "r") as file:
        data = json.load(file)
        
    sentences = data["sentences"]
    embeddings = data["embeddings"]
    
    query_embedding = model.encode([query])[0]
    
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    sorted_indices = np.argsort(similarities)[::-1]
    top_indices = sorted_indices[:top_n]
    
    results = [(sentences[idx], similarities[idx]) for idx in top_indices]
    
    return results


def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a**2 for a in vec1) ** 0.5
    magnitude2 = sum(b**2 for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)


if __name__ == "__main__":
    with open("test_text", 'r') as file:
        test_text = file.read()

    filename = "embeddings.json"
    encode_and_save_conversation_from_file("conversations/meditation_notes.json", filename)

    query = "How can I sort an array in Python?"
    top_matches = retrieve_sentences(query, filename, top_n=5)
    for sentence, similarity in top_matches:
        print(f"Sentence: {sentence}")
        print(f"Similarity: {similarity}")
        print("-----------------------------------")
    
