import click

from longchat import AnthropicClient

client = AnthropicClient("anthropic_key")

# Claude can handle 200 000 tokens, which is about 100 000 words. In practice, it's better
# to keep the window slightly smaller.
chunk_size = 5000


def combine_chunks(chunks):
    i = 0
    while i < len(chunks) - 1:
        if len(chunks[i].split(" ")) + len(chunks[i+1].split(" ")) < chunk_size - 10:
            chunks[i] += "\n\n" + chunks[i+1]
            chunks.pop(i+1)
        else:
            i += 1
    return chunks


def split_by(text, separator="\n\n\n", reverse=False):
    chunks = text.split(separator)
    if reverse:
        chunks = chunks[::-1]
    sections = []

    for chunk in chunks:
        if len(chunk.split(" ")) > chunk_size:
            if separator == "\n\n\n":
                sections.extend(chunk.split("\n\n"))
            elif separator == "\n\n":
                sections.extend(chunk.split("\n"))
        else:
            sections.append(chunk)

    #combine chunks if smaller than chunk_size
    sections = combine_chunks(sections)
    return sections


def running_summary(filename, reverse, request_text):
    """ Summarizes each chunk and provides concatenates the summary to the next chunk. This naturally emphasizes newer content in the text."""
    with open(filename, "r") as f:
        text = f.read()
    
    # Start with each query with a simple system message, then a request message from the user
    system_message = {"role": "system", "content": "You are a helpful AI assistant. You help users by summarizing texts and providing insights about them."}
    request_text = f"\n\n{request_text}\n\n"
    summary = "(No summary yet, first paragraph)"

    # First split the text into paragraphs. Then compine into up to 3000 word chunks.
    sections = split_by(text, reverse=reverse)
    for i, section in enumerate(sections):
        print(f"Lenght of section {i}: {len(section.split(" "))}")
        
        messages = [
            system_message,
            {"role": "user", "content": summary + request_text + section}
        ]

        # Get the response from the AI
        summary = "summary: " + client.request_message(messages)
        print(summary)


@click.command()
@click.option("--filename")
@click.option("--reverse", is_flag=True)
@click.option("--initial_request", default="Please summarize the following text.")
@click.option("--combination_request", default="Please combine the following summaries into a single summary.")
def summarize(filename, reverse, initial_request, combination_request):
    """ Hierarchical summary. Each chunk is summarized separately. Summaries are then concatenated and summarized. Logarithmic time, does not emphasize newer content."""
    with open(filename, "r") as f:
        text = f.read()
    
    # Start with each query with a simple system message, then a request message from the user
    system_message = {"role": "system", "content": "You are a helpful AI assistant. You help users by summarizing texts and providing insights about them."}
    request_text = f"{initial_request}\n\n"

    # First split the text into paragraphs. Then compine into up to 3000 word chunks.
    sections = split_by(text, reverse=reverse)

    while len(sections) > 1:
        summaries = []
        for i, section in enumerate(sections):
            print(f"Lenght of section {i}/{len(sections)}: {len(sections[i].split(" "))}")
        
            messages = [
                system_message,
                {"role": "user", "content":  request_text + sections[i]}
            ]

            # Get the response from the AI
            summaries.append(client.request_message(messages))
            print(f"length of summary {i}/{len(sections)}: {len(summaries[i].split(' '))}")
        
        summaries = combine_chunks(summaries)
        print(f"Summarized {len(sections)} sections into {len(summaries)} sections")

        request_text = f"{combination_request}\n\n"
        sections = summaries
    
    # Finally summarize the single concatenated section
    messages = [
        system_message,
        {"role": "user", "content": request_text + sections[0]}
    ]
    summary = client.request_message(messages)

    print("----------------")
    print(summary)



if __name__ == "__main__":
    summarize()

    
    