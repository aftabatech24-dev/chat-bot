"""Lightweight intent classification for messages not covered by AIML rules."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
import re
from typing import Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:  # Keep the chatbot usable when optional ML wheels are unavailable.
    TfidfVectorizer = None
    MultinomialNB = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False


@dataclass(frozen=True)
class TrainingExample:
    intent: str
    text: str


@dataclass(frozen=True)
class IntentPrediction:
    intent: str
    confidence: float
    similarity: float


_TRAINING_EXAMPLES = [
    TrainingExample("greeting", "hello"),
    TrainingExample("greeting", "hi there"),
    TrainingExample("greeting", "hey"),
    TrainingExample("greeting", "howdy"),
    TrainingExample("greeting", "what's up"),
    TrainingExample("greeting", "nice to meet you"),
    TrainingExample("greeting", "good afternoon"),
    TrainingExample("greeting", "good evening"),
    TrainingExample("how_are_you", "how are you"),
    TrainingExample("how_are_you", "how are you doing today"),
    TrainingExample("how_are_you", "how is it going"),
    TrainingExample("how_are_you", "how has your day been"),
    TrainingExample("how_are_you", "are you doing okay"),
    TrainingExample("mood_positive", "I am doing well"),
    TrainingExample("mood_positive", "I feel great today"),
    TrainingExample("mood_positive", "I had a good day"),
    TrainingExample("mood_positive", "things are going well"),
    TrainingExample("mood_positive", "I am happy"),
    TrainingExample("mood_negative", "I am feeling sad"),
    TrainingExample("mood_negative", "I had a rough day"),
    TrainingExample("mood_negative", "I feel stressed out"),
    TrainingExample("mood_negative", "I feel lonely"),
    TrainingExample("mood_negative", "I am worried about something"),
    TrainingExample("thanks", "thank you"),
    TrainingExample("thanks", "thanks a lot"),
    TrainingExample("thanks", "that was helpful"),
    TrainingExample("thanks", "I appreciate your help"),
    TrainingExample("thanks", "many thanks"),
    TrainingExample("goodbye", "goodbye"),
    TrainingExample("goodbye", "bye see you later"),
    TrainingExample("goodbye", "I have to go"),
    TrainingExample("goodbye", "talk to you later"),
    TrainingExample("goodbye", "good night"),
    TrainingExample("identity", "what is your name"),
    TrainingExample("identity", "who are you"),
    TrainingExample("identity", "tell me about yourself"),
    TrainingExample("identity", "who created you"),
    TrainingExample("identity", "are you a real person"),
    TrainingExample("identity", "are you a robot"),
    TrainingExample("identity", "do you have feelings"),
    TrainingExample("identity", "what is your favorite color"),
    TrainingExample("artificial_intelligence", "what is artificial intelligence"),
    TrainingExample("artificial_intelligence", "explain AI to me"),
    TrainingExample("artificial_intelligence", "how does artificial intelligence work"),
    TrainingExample("artificial_intelligence", "tell me about machine learning"),
    TrainingExample("artificial_intelligence", "what does NLP mean"),
    TrainingExample("weather", "what is the weather"),
    TrainingExample("weather", "will it rain today"),
    TrainingExample("weather", "is it sunny outside"),
    TrainingExample("weather", "what is the forecast for my city"),
    TrainingExample("weather", "is it cold outside"),
    TrainingExample("capabilities", "what can you do"),
    TrainingExample("capabilities", "how can you help me"),
    TrainingExample("capabilities", "what topics can I ask about"),
    TrainingExample("capabilities", "can you save a note for me"),
    TrainingExample("joke", "tell me a joke"),
    TrainingExample("joke", "make me laugh"),
    TrainingExample("joke", "do you know any funny jokes"),
    TrainingExample("joke", "say something funny"),
    TrainingExample("apology", "I am sorry"),
    TrainingExample("apology", "sorry about that"),
    TrainingExample("apology", "please forgive me"),
    TrainingExample("agreement", "yes that sounds good"),
    TrainingExample("agreement", "okay sure"),
    TrainingExample("agreement", "I agree with you"),
    TrainingExample("agreement", "that works for me"),
    TrainingExample("disagreement", "no thank you"),
    TrainingExample("disagreement", "I do not think so"),
    TrainingExample("disagreement", "not really"),
    TrainingExample("disagreement", "I disagree"),
    TrainingExample("compliment", "you are helpful"),
    TrainingExample("compliment", "you are smart"),
    TrainingExample("compliment", "good job chatbot"),
    TrainingExample("compliment", "you are nice"),
    TrainingExample("small_talk", "I like listening to music"),
    TrainingExample("small_talk", "I enjoy reading books"),
    TrainingExample("small_talk", "that is interesting"),
    TrainingExample("small_talk", "tell me something interesting"),
    TrainingExample("small_talk", "I feel a little tired"),
]

_RESPONSES = {
    "greeting": [
        "Hello! What would you like to talk about?",
        "Hi there! How can I help?",
        "Hey! What is on your mind?",
    ],
    "how_are_you": [
        "I'm doing well and ready to chat. How are you?",
        "Thanks for asking! I'm here and ready to help. How is your day going?",
    ],
    "mood_positive": [
        "That's great to hear! What has been going well?",
        "I'm glad things are going well. Want to tell me more?",
    ],
    "mood_negative": [
        "I'm sorry things feel difficult right now. Want to tell me a little more?",
        "That sounds tough. I'm here to listen if you'd like to talk about it.",
    ],
    "thanks": ["You're welcome!", "Glad I could help."],
    "goodbye": ["Goodbye! Take care.", "See you next time!"],
    "identity": [
        "I'm a chatbot built for friendly conversation. I use AIML rules and a small intent classifier, so I can make mistakes.",
        "I'm an AI chatbot, not a person. I don't have feelings or personal experiences, but I can still chat and try to help.",
    ],
    "artificial_intelligence": [
        "Artificial intelligence is the field of building computer systems that can perform tasks such as understanding language, recognizing patterns, and making decisions."
    ],
    "weather": [
        "I don't have a live weather service connected, so I can't check current conditions. If you share a forecast, I can help explain it."
    ],
    "capabilities": [
        "I can chat about everyday topics, answer a few common questions, explain basic AI, and help you save notes. I don't have live web or weather access."
    ],
    "joke": [
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "Why don't scientists trust atoms? Because they make up everything!",
    ],
    "apology": ["No worries. We can keep going. What would you like to talk about?"],
    "agreement": ["Sounds good!", "Okay, let's do that."],
    "disagreement": ["No problem. What would work better for you?", "That's okay. We can try another approach."],
    "compliment": ["Thanks, that's kind of you to say! What would you like to talk about?"],
    "small_talk": [
        "That sounds nice. What do you enjoy most about it?",
        "Interesting! Tell me a little more.",
        "Thanks for sharing. What else is on your mind?",
    ],
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").casefold())


class IntentClassifier:
    """TF-IDF + Naive Bayes classifier with cosine and pure-Python fallback."""

    def __init__(self, confidence_threshold: float = 0.28, similarity_threshold: float = 0.16) -> None:
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.examples = _TRAINING_EXAMPLES
        self.texts = [example.text for example in self.examples]
        self.labels = [example.intent for example in self.examples]
        self.vectorizer = None
        self.model = None
        self.matrix = None
        self._idf: dict[str, float] = {}
        self._fallback_vectors: list[dict[str, float]] = []

        if SKLEARN_AVAILABLE:
            try:
                self.vectorizer = TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                )
                self.matrix = self.vectorizer.fit_transform(self.texts)
                self.model = MultinomialNB(alpha=0.25).fit(self.matrix, self.labels)
            except (ValueError, RuntimeError):
                self.vectorizer = self.model = self.matrix = None

        if self.model is None:
            self._build_python_tfidf()

    def _build_python_tfidf(self) -> None:
        documents = [set(_tokens(text)) for text in self.texts]
        document_frequency: Counter[str] = Counter(
            token for document in documents for token in document
        )
        count = len(documents)
        self._idf = {
            token: math.log((1 + count) / (1 + frequency)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self._fallback_vectors = [self._vectorize_python(text) for text in self.texts]

    def _vectorize_python(self, text: str) -> dict[str, float]:
        term_counts = Counter(_tokens(text))
        vector = {
            token: (1.0 + math.log(frequency)) * self._idf.get(token, 0.0)
            for token, frequency in term_counts.items()
            if token in self._idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            vector = {token: value / norm for token, value in vector.items()}
        return vector

    @staticmethod
    def _sparse_cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    def predict_intent(self, text: str) -> Optional[IntentPrediction]:
        normalized = (text or "").strip()
        if not normalized:
            return None

        if self.model is not None and self.vectorizer is not None and self.matrix is not None:
            vector = self.vectorizer.transform([normalized])
            if vector.nnz == 0:
                return None
            probabilities = self.model.predict_proba(vector)[0]
            class_probabilities = dict(zip(self.model.classes_, probabilities))
            similarities = cosine_similarity(vector, self.matrix).ravel()
            intent_similarities: dict[str, float] = defaultdict(float)
            for label, similarity in zip(self.labels, similarities):
                intent_similarities[label] = max(
                    intent_similarities[label], float(similarity)
                )
            intent = max(
                intent_similarities,
                key=lambda label: (
                    0.55 * class_probabilities.get(label, 0.0)
                    + 0.45 * intent_similarities[label]
                ),
            )
            confidence = float(class_probabilities.get(intent, 0.0))
            similarity = intent_similarities[intent]
        else:
            vector = self._vectorize_python(normalized)
            if not vector:
                return None
            similarities = [
                self._sparse_cosine(vector, sample)
                for sample in self._fallback_vectors
            ]
            best_index = max(range(len(similarities)), key=similarities.__getitem__)
            intent = self.labels[best_index]
            similarity = similarities[best_index]
            matches = [
                score for label, score in zip(self.labels, similarities) if label == intent
            ]
            confidence = max(matches) / (sum(matches) + 1e-9)

        if confidence < self.confidence_threshold or similarity < self.similarity_threshold:
            return None
        return IntentPrediction(intent, confidence, similarity)

    def respond(self, text: str) -> str:
        prediction = self.predict_intent(text)
        if prediction is None:
            return (
                "I'm not sure I understood that. Try asking me about AI, the weather, "
                "or what I can do."
            )
        responses = _RESPONSES.get(prediction.intent)
        if not responses:
            return "I'm still learning how to answer that. Could you try asking another way?"
        return random.choice(responses)


_DEFAULT_CLASSIFIER = IntentClassifier()


def predict_intent(text: str) -> Optional[IntentPrediction]:
    """Convenience API for callers that only need the predicted intent."""
    return _DEFAULT_CLASSIFIER.predict_intent(text)


def get_response(text: str) -> str:
    """Return a natural-language response for an unhandled user message."""
    return _DEFAULT_CLASSIFIER.respond(text)
