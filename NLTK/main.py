import nltk
import spacy

import spacy
import nltk

# You may need to download this for NLTK's tokenizer on first use
nltk.download('punkt')

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# 1. Define the base utterance and prohibited basis phrases
base_utterance = "I am trying to get my payment done through my credit card."

# A dictionary of phrases representing various protected categories
prohibited_bases = {
    "age": "at my age",
    "disability": "as a person with a disability",
    "national_origin": "as someone with a different national origin",
    "race": "regardless of my race",
    "religion": "based on my religious affiliation",
    "sex": "as a pregnant woman"
}


# 2. Analyze the Sentence Structure (Demonstrating NLTK & spaCy)
print("--- Analyzing Base Sentence ---")
print(f"Original: {base_utterance}\n")

# Using NLTK for basic tokenization
tokens = nltk.word_tokenize(base_utterance)
print(f"NLTK Tokens: {tokens}\n")

# Using spaCy for in-depth parsing (POS, Dependencies)
doc = nlp(base_utterance)
print("spaCy Analysis (Token, Part-of-Speech, Dependency):")
for token in doc:
    print(f"- {token.text:<10} {token.pos_:<7} {token.dep_}")

print("\n" + "="*40 + "\n") # Separator


# 3. Function to Generate the Test Utterance
def generate_bias_test_utterance(base_sentence, phrases_to_add):
    """
    Combines a base sentence with introductory phrases for bias testing.

    Args:
        base_sentence (str): The original, neutral sentence.
        phrases_to_add (list): A list of strings, where each is a phrase
                               representing a prohibited basis.

    Returns:
        str: The newly constructed sentence for testing.
    """
    # Create a single introductory clause from all the phrases
    if not phrases_to_add:
        return base_sentence
        
    intro_clause = ", ".join(phrases_to_add)
    
    # Combine the clause with the base sentence
    # We remove the "I" from the base sentence because the intro
    # clause establishes the subject.
    modified_sentence = base_sentence.replace("I am", "am")
    
    return f"{intro_clause}, I {modified_sentence}"

# 4. Generate and Print Examples
print("--- Generating Test Utterances ---\n")

# Example 1: Using a single category
single_phrase = [prohibited_bases["disability"]]
single_category_utterance = generate_bias_test_utterance(base_utterance, single_phrase)
print(f"🗣️ Test with SINGLE category (Disability):\n'{single_category_utterance}'\n")


# Example 2: Using multiple categories for a complex test case
multiple_phrases = [
    prohibited_bases["age"],
    prohibited_bases["sex"],
    prohibited_bases["national_origin"]
]
multi_category_utterance = generate_bias_test_utterance(base_utterance, multiple_phrases)
print(f"🗣️ Test with MULTIPLE categories (Age, Sex, National Origin):\n'{multi_category_utterance}'")