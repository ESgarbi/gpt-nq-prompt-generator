import random
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')


prefixes = ["Senior", "Junior", "Lead", "Assistant", "Associate"]

def modify_job_title(job_title):
    if random.choice([True, False]):
        prefix = random.choice(prefixes)
        return f"{prefix} {job_title}"
    return job_title


def introduce_typos(text, probability=0.05):
    words = text.split()
    for i, word in enumerate(words):
        if random.random() < probability:
            typo_position = random.randint(0, len(word)-1)
            action = random.choice(["remove", "replace", "add"])
            if action == "remove" and len(word) > 1:
                word = word[:typo_position] + word[typo_position+1:]
            elif action == "replace":
                replacement = random.choice('abcdefghijklmnopqrstuvwxyz')
                word = word[:typo_position] + replacement + word[typo_position+1:]
            elif action == "add":
                addition = random.choice('abcdefghijklmnopqrstuvwxyz')
                word = word[:typo_position] + addition + word[typo_position:]
            words[i] = word
    return " ".join(words)

def swap_sentences(text):
    sentences = text.split(".")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if len(sentences) < 2:
        return text
    idx1, idx2 = random.sample(range(len(sentences)), 2)
    sentences[idx1], sentences[idx2] = sentences[idx2], sentences[idx1]
    return ". ".join(sentences)


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))  # Replace underscores with spaces for multi-word synonyms... Debug into a sample word to see if this is necessary.. sample it... dont loose sleep over it.
    return list(synonyms)

def replace_with_synonyms(text, probability=0.2):
    words = text.split()
    for i, word in enumerate(words):
        fetched_synonyms = get_synonyms(word)
        relevant_synonyms = list(set(fetched_synonyms))
        if relevant_synonyms and random.random() < probability:
            synonym = random.choice(relevant_synonyms)
            words[i] = synonym
    return " ".join(words)

def generate_suggestion_request(job_title):
    # todo remove before publishing code
    # call out to llm api (get key etc...)
    pass

def get_question_context(title):
    modified_title = modify_job_title(title)
    

    original_description = f"I want you to act as a {modified_title}. Your main responsibilities include tasks related to {modified_title}."

    description = introduce_typos(original_description)
    description = swap_sentences(description)
    description = replace_with_synonyms(description)
    
    suggestion_request = generate_suggestion_request(modified_title)
    
    return modified_title, original_description, description, suggestion_request


def generate_batch_synthetic_data(job_titles, batch_size):
    data = []
    for job_title in job_titles:
        modified_title = modify_job_title(job_title)
        description = f"I want you to act as a {modified_title}. Your main responsibilities include tasks related to {modified_title}."
        description = introduce_typos(description)
        description = swap_sentences(description)
        description = replace_with_synonyms(description)
        suggestion_request = generate_suggestion_request(modified_title)
        data.append((modified_title, description, suggestion_request))
    return data