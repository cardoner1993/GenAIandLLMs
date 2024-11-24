import argparse
from transformers import MarianMTModel, MarianTokenizer # Require pip install sentencepiece
# Warning avoid with pip install sacremoses

# IBM specialization course. Creating a default translator from source lang to target lang.

def translate(text, source_lang, target_lang):
    # Define the model for the language pair
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize and translate
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Avoid Warnings export TOKENIZERS_PARALLELISM=false

def chat_with_bot(source_lang="es", target_lang="it"):
        print("Starting Conversation")
        while True:
                # Get User Input
                input_text = input("You: ")

                # Exit conditions.
                if input_text.lower() in ["quit", "exit", "bye", "adios", "arrivederci"]:
                        print("Chatbot: Goodbye!!!!!")
                        break

                # Tokenize input and generate response.
                # Tokenize and summarize
                translated_text = translate(input_text, source_lang, target_lang)

                print("Chatbot Translation:", translated_text)


if __name__ == "__main__":
        # Parse command-line arguments for languages
        parser = argparse.ArgumentParser(description="Specify source and target languages for translation.")
        parser.add_argument("source_lang", type=str, help="Source language code (e.g., 'es' for Spanish).")
        parser.add_argument("target_lang", type=str, help="Target language code (e.g., 'it' for Italian).")
        args = parser.parse_args()

        # Assign the source and target languages from arguments
        source_lang = args.source_lang
        target_lang = args.target_lang
        chat_with_bot(source_lang, target_lang)