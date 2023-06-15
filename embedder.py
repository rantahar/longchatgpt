from sentence_transformers import SentenceTransformer
import nltk
import json
import numpy as np
import tqdm

# Load the pre-trained SBERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')
nltk.download('punkt')


def encode_sentences(text):
    sentences = nltk.sent_tokenize(text)
    embeddings = model.encode(sentences)
    return sentences, embeddings


def encode_and_save_sentences(text, filename):
    sentences, embeddings = encode_sentences(text)
    
    data = {"sentences": sentences, "embeddings": embeddings.tolist()}
    
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


def encode_and_append(text, filename):
    with open(filename, "r") as file:
        data = json.load(file)
    sentences = data["sentences"]
    embeddings = data["embeddings"]

    text = clean_message(text)
    _sentences, _embeddings = encode_sentences(text)
    if _embeddings.ndim < 2:
        return
    sentences += _sentences
    embeddings = np.concatenate([embeddings, _embeddings], axis=0)

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
            if _embeddings.ndim < 2:
                continue
            sentences += _sentences
            if embeddings is not None:
                embeddings = np.concatenate([embeddings, _embeddings], axis=0)
            else:
                embeddings = _embeddings
    
    data = {"sentences": sentences, "embeddings": embeddings.tolist()}
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
    
    similarities = []
    for embedding in embeddings:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append(similarity)
    
    sorted_indices = sorted(range(len(similarities)), key=lambda k: similarities[k], reverse=True)
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
    
