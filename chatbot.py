"""Train and run a small Naive Bayes intent chatbot.

Keep this file beside chatbot_data.json. Run with: python chatbot.py
Only Python's standard library is used.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re


DATA_FILE = Path(__file__).with_name("chatbot_data.json")
TEST_FRACTION = 0.20
RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.12


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def extract_features(text):
    """Return word and adjacent-word features for the classifier."""
    words = tokenize(text)
    word_features = ["word:" + word for word in words]
    phrase_features = [
        "pair:" + first + "_" + second
        for first, second in zip(words, words[1:])
    ]
    return word_features + phrase_features


def load_dataset(path=DATA_FILE):
    """Read and check the labelled examples and replies in the JSON file."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(data.get("intents"), list):
        raise ValueError("The dataset must contain an 'intents' list.")

    examples = []
    responses = {}
    for intent in data["intents"]:
        if not isinstance(intent, dict):
            raise ValueError("Each intent in the dataset must be an object.")

        tag = intent.get("tag")
        phrases = intent.get("examples")
        replies = intent.get("responses")
        if not tag or not isinstance(phrases, list) or not phrases:
            raise ValueError("Every intent needs a tag and at least one example.")
        if tag in responses:
            raise ValueError("Intent tags must be unique: " + tag)
        if not isinstance(replies, list) or not replies:
            raise ValueError("Every intent needs at least one response: " + tag)

        responses[tag] = replies
        examples.extend((phrase, tag) for phrase in phrases if isinstance(phrase, str) and phrase.strip())

    if len(responses) < 2:
        raise ValueError("The dataset needs at least two different intents.")
    if any(not any(label == tag for _, label in examples) for tag in responses):
        raise ValueError("Every intent must have at least one valid example.")
    return examples, responses


def split_dataset(examples, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
    """Create a repeatable, stratified train/test split for every intent."""
    examples_by_intent = defaultdict(list)
    for text, label in examples:
        examples_by_intent[label].append((text, label))

    rng = random.Random(seed)
    training_examples = []
    test_examples = []
    for label, intent_examples in sorted(examples_by_intent.items()):
        if len(intent_examples) < 2:
            raise ValueError("Each intent needs at least two examples for testing: " + label)
        shuffled = list(intent_examples)
        rng.shuffle(shuffled)
        test_count = max(1, int(len(shuffled) * test_fraction))
        test_count = min(test_count, len(shuffled) - 1)
        test_examples.extend(shuffled[:test_count])
        training_examples.extend(shuffled[test_count:])

    return training_examples, test_examples


class NaiveBayesClassifier:
    """A small Multinomial Naive Bayes classifier implemented in Python."""

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.labels = []
        self.vocabulary = set()
        self.log_priors = {}
        self.word_counts = {}
        self.total_word_counts = {}

    def fit(self, training_examples):
        """Learn intent priors and word frequencies from labelled examples."""
        document_counts = Counter()
        word_counts = defaultdict(Counter)

        for text, label in training_examples:
            document_counts[label] += 1
            features = extract_features(text)
            word_counts[label].update(features)
            self.vocabulary.update(features)

        if not document_counts or not self.vocabulary:
            raise ValueError("The training data must contain words and intent labels.")

        total_documents = sum(document_counts.values())
        self.labels = sorted(document_counts)
        self.log_priors = {
            label: math.log(document_counts[label] / total_documents)
            for label in self.labels
        }
        self.word_counts = {label: word_counts[label] for label in self.labels}
        self.total_word_counts = {
            label: sum(word_counts[label].values()) for label in self.labels
        }

    def predict_probabilities(self, text):
        """Return the estimated probability for every intent label."""
        features = [feature for feature in extract_features(text) if feature in self.vocabulary]
        if not features:
            return {}

        feature_frequencies = Counter(features)
        vocabulary_size = len(self.vocabulary)
        log_scores = {}

        for label in self.labels:
            score = self.log_priors[label]
            denominator = self.total_word_counts[label] + self.smoothing * vocabulary_size
            for feature, frequency in feature_frequencies.items():
                numerator = self.word_counts[label][feature] + self.smoothing
                score += frequency * math.log(numerator / denominator)
            log_scores[label] = score

        # Convert log scores to probabilities without numerical underflow.
        largest_score = max(log_scores.values())
        score_total = sum(math.exp(score - largest_score) for score in log_scores.values())
        normalizer = largest_score + math.log(score_total)
        return {label: math.exp(score - normalizer) for label, score in log_scores.items()}

    def predict(self, text):
        """Return the best label and its probability, or no prediction."""
        probabilities = self.predict_probabilities(text)
        if not probabilities:
            return None, 0.0
        return max(probabilities.items(), key=lambda item: item[1])


def measure_accuracy(classifier, test_examples):
    """Measure correct predictions on examples withheld from training."""
    if not test_examples:
        return 0.0
    correct = sum(classifier.predict(text)[0] == label for text, label in test_examples)
    return correct / len(test_examples)


class Chatbot:
    """Load the data, evaluate a split, then train the final intent model."""

    def __init__(self, dataset_path=DATA_FILE):
        examples, self.responses = load_dataset(dataset_path)
        training_examples, test_examples = split_dataset(examples)

        # First fit only on the training portion to get a fair test measurement.
        evaluation_model = NaiveBayesClassifier()
        evaluation_model.fit(training_examples)
        self.test_accuracy = measure_accuracy(evaluation_model, test_examples)
        self.training_example_count = len(training_examples)
        self.test_example_count = len(test_examples)
        self.example_count = len(examples)
        self.intent_count = len(self.responses)

        # After evaluation, train the chatbot's final model on all available data.
        self.classifier = NaiveBayesClassifier()
        self.classifier.fit(examples)

    def respond(self, message):
        """Predict an intent and return a prepared reply or a fallback."""
        intent, confidence = self.classifier.predict(message)
        if intent is None or confidence < CONFIDENCE_THRESHOLD:
            return "I'm not confident about that message yet. Could you say it another way?"
        return random.choice(self.responses[intent])


def main():
    """Start the terminal chat and show the held-out evaluation result."""
    try:
        bot = Chatbot()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print("Could not start the chatbot:", error)
        print("Keep chatbot.py and chatbot_data.json in the same folder.")
        return

    print("=" * 54)
    print("  PYTHON CHATBOT - NAIVE BAYES INTENT CLASSIFIER")
    print("  Dataset: {} examples across {} intents".format(bot.example_count, bot.intent_count))
    print("  Train/test: {} / {} examples".format(
        bot.training_example_count, bot.test_example_count
    ))
    print("  Held-out test accuracy: {:.1%}".format(bot.test_accuracy))
    print("  Type 'help' for a tip or 'exit' to quit.")
    print("=" * 54)

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if not message:
            continue
        if message.lower() in {"exit", "quit"}:
            print("Bot: Goodbye! Thanks for chatting.")
            break
        if message.lower() == "help":
            print("Bot: Try a greeting, ask 'what is AI?', or tell me how you feel.")
            continue

        print("Bot:", bot.respond(message))


if __name__ == "__main__":
    main()
