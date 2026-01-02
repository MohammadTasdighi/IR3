from hazm import Normalizer, Lemmatizer

normalizer = Normalizer()
lemmatizer = Lemmatizer()

def clean_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = normalizer.normalize(text)
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(lemmatized_words)