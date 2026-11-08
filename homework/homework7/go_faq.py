"""Справочник Go: поиск, оценка качества и консольный диалог."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import groupby
import json
from math import log, sqrt
from pathlib import Path
from random import choice
import re

from nltk.stem.snowball import RussianStemmer


ROOT = Path(__file__).resolve().parent
KNOWLEDGE = ROOT / "go_knowledge.json"
PERSONAL = ROOT / "go_knowledge.user.json"
QUESTIONS = ROOT / "go_questions.json"
NO_ANSWER = "В базе нет достаточно близкого ответа."
METRICS = ("Жаккар", "Косинус", "Пирсон", "Спирмен")
AUTO = "Автовыбор"
SHORT_LIMIT = 3
STOP = set("""
а без бы в во вот все где для до его ее если есть же за и из или им их к
как какая какие какой каким когда кто ли меж между мне можно мы на над не
него нее нет но о об один он она они от по под при про с со так такой то
у уже чего чем что чтобы эта это этот является зачем нужно надо
the a an to from and or in of for is it on with
""".split())
TOKEN_RE = re.compile(r"[а-я]+|[a-z][a-z0-9]*|%w")
STEMMER = RussianStemmer()


def tokenize(text):
    """Стемминг только для русских слов; имена API остаются токенами."""
    words = TOKEN_RE.findall(text.casefold().replace("ё", "е"))
    return [
        STEMMER.stem(word) if "а" <= word[0] <= "я" else word
        for word in words
        if word not in STOP
    ]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit(vector, center=False):
    """Нормированный вектор; для корреляции предварительно вычитается среднее."""
    if center and vector:
        mean = sum(vector) / len(vector)
        vector = [x - mean for x in vector]
    norm = sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else [0.0] * len(vector)


def ranks(vector):
    """Средний ранг при одинаковых значениях, включая нули."""
    result = [0.0] * len(vector)
    order = sorted(range(len(vector)), key=vector.__getitem__)
    start = 1
    for _, group in groupby(order, key=vector.__getitem__):
        indices = list(group)
        average = start + (len(indices) - 1) / 2
        for i in indices:
            result[i] = average
        start += len(indices)
    return result


def dot(left, right):
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(left, right))))


def similarity(metric, left_set, right_set, left_vector, right_vector):
    """Косинус и корреляции считаются скалярным произведением нормированных векторов."""
    if metric == "Жаккар":
        union = left_set | right_set
        return len(left_set & right_set) / len(union) if union else 0.0
    return dot(left_vector, right_vector)


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    matched_question: str


class Search:
    def __init__(self, entries, representation="TF-IDF", variants=True):
        self.representation = representation
        self.entries = {entry["id"]: entry for entry in entries}
        if not entries or len(self.entries) != len(entries):
            raise ValueError("Нужны записи с уникальными id.")
        documents = []
        self.forms = {}
        for entry in entries:
            questions = [entry["question"], *entry["variants"]] if variants else [entry["question"]]
            for question in questions:
                tokens = tuple(tokenize(question))
                if not tokens:
                    raise ValueError("Формулировка пуста после предобработки.")
                if tokens in self.forms and self.forms[tokens] != entry["id"]:
                    raise ValueError(f"Одинаковая формулировка у разных записей: {question}")
                self.forms[tokens] = entry["id"]
                documents.append((entry["id"], question, Counter(tokens)))
        self.df = Counter(word for _, _, counts in documents for word in counts)
        self.vocabulary = sorted(self.df)
        self.idf = {
            word: log((1 + len(documents)) / (1 + df)) + 1
            for word, df in self.df.items()
        }
        self.documents = [
            (entry_id, question, set(counts), self.prepare(counts))
            for entry_id, question, counts in documents
        ]

    def vectorize(self, counts):
        if self.representation == "BoW":
            return [counts.get(word, 0) for word in self.vocabulary]
        return [counts.get(word, 0) * self.idf[word] for word in self.vocabulary]

    def prepare(self, counts):
        vector = self.vectorize(counts)
        return {
            "Косинус": unit(vector),
            "Пирсон": unit(vector, center=True),
            "Спирмен": unit(ranks(vector), center=True),
        }

    def rank(self, question):
        tokens = tokenize(question)
        counts = Counter(tokens)
        query_words = set(counts).intersection(self.df)
        best = {metric: {} for metric in METRICS}
        if query_words:
            query_vectors = self.prepare(counts)
            for entry_id, form, words, vectors in self.documents:
                if not (query_words & words):
                    continue
                for metric in METRICS:
                    score = similarity(metric, query_words, words, query_vectors.get(metric), vectors.get(metric))
                    old = best[metric].get(entry_id)
                    if old is None or score > old.score:
                        best[metric][entry_id] = Hit(entry_id, score, form)
        return len(tokens), {
            metric: sorted(hits.values(), key=lambda hit: (-hit.score, hit.id))
            for metric, hits in best.items()
        }


def choose(hits, threshold=0.0):
    return hits[0].id if hits and hits[0].score >= threshold else None


def fraction(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate(expected, predictions, rankings):
    """Top-3 — среди вопросов с эталонным ответом, до применения порога."""
    found = sum(want is not None and want == got for want, got in zip(expected, predictions))
    known = sum(want is not None for want in expected)
    issued = sum(got is not None for got in predictions)
    correct = sum(want == got for want, got in zip(expected, predictions))
    rejected = sum(want is None and got is None for want, got in zip(expected, predictions))
    top3 = sum(
        want is not None and want in {hit.id for hit in hits[:3]}
        for want, hits in zip(expected, rankings)
    )
    return {
        "Accuracy": fraction(correct, len(expected)),
        "Top-3 accuracy": fraction(top3, known),
        "Precision": fraction(found, issued),
        "Recall": fraction(found, known),
        "Отказы вне базы": fraction(rejected, len(expected) - known),
        "Верных": correct,
    }


def minimum_scores(validation, expected):
    """Подбор минимального сходства отдельно для каждой метрики на validation."""
    thresholds = {}
    for metric in METRICS:
        rankings = [ranked[metric] for _, ranked in validation]
        values = sorted({hits[0].score for hits in rankings if hits and hits[0].score > 0})
        candidates = [0.000001, 1.000001]
        candidates += [(a + b) / 2 for a, b in zip([0.0, *values], [*values, 1.000002])]

        def quality(threshold):
            pred = [choose(hits, threshold) for hits in rankings]
            stat = evaluate(expected, pred, rankings)
            balanced = (stat["Recall"] + stat["Отказы вне базы"]) / 2
            return balanced, stat["Accuracy"], stat["Precision"], threshold

        thresholds[metric] = max(candidates, key=quality)
    return thresholds


def calibrate_adaptive(validation, expected, thresholds):
    """Доля верных ответов среди выданных по каждой метрике и длине."""
    reliability = {}
    for short in (True, False):
        for metric in METRICS:
            accepted = [
                (want, choose(ranked[metric], thresholds[metric]))
                for (length, ranked), want in zip(validation, expected)
                if (length <= SHORT_LIMIT) == short
            ]
            correct = sum(got is not None and got == want for want, got in accepted)
            issued = sum(got is not None for _, got in accepted)
            reliability[short, metric] = (correct + 1) / (issued + 2)
    return reliability


def adaptive_metric(length, ranked, thresholds, reliability):
    short = length <= SHORT_LIMIT
    preferred = "Жаккар" if short else "Косинус"
    accepted = {
        metric: hits[0]
        for metric, hits in ranked.items()
        if choose(hits, thresholds[metric]) is not None
    }
    if not accepted:
        return preferred
    if preferred in accepted and len({hit.id for hit in accepted.values()}) == 1:
        return preferred

    def confidence(metric):
        hits = ranked[metric]
        first = hits[0].score
        second = hits[1].score if len(hits) > 1 else 0
        threshold = thresholds[metric]
        excess = fraction(first - threshold, max(1e-9, 1 - threshold))
        gap = min(1, first - second)
        return 0.6 * reliability[short, metric] + 0.25 * excess + 0.15 * gap

    return max(accepted, key=confidence)


def fit_dialogue(entries, validation):
    engine = Search(entries)
    expected = [want for _, want in validation]
    ranked = [engine.rank(question) for question, _ in validation]
    thresholds = minimum_scores(ranked, expected)
    return engine, thresholds, calibrate_adaptive(ranked, expected, thresholds)


def learn(entries, kind, entry_id, text, answer=""):
    """Добавить новую пару, парафраз или альтернативный ответ и проверить индекс."""
    updated = deepcopy(entries)
    by_id = {entry["id"]: entry for entry in updated}
    entry_id, text, answer = entry_id.strip(), text.strip(), answer.strip()
    if not entry_id or not text:
        raise ValueError("ID и текст должны быть непустыми.")
    if kind == "new":
        if entry_id in by_id or not answer:
            raise ValueError("Нужны новый ID и непустой ответ.")
        updated.append({"id": entry_id, "question": text, "variants": [], "answers": [answer]})
    elif kind in {"variant", "answer"}:
        if entry_id not in by_id:
            raise ValueError(f"Неизвестный ID: {entry_id}")
        entry = by_id[entry_id]
        field = "variants" if kind == "variant" else "answers"
        if text not in entry[field] and (kind != "variant" or text != entry["question"]):
            entry[field].append(text)
    else:
        raise ValueError("Неизвестная операция.")
    Search(updated)
    return updated


def save_personal(entries):
    temporary = PERSONAL.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(PERSONAL)


HELP = """
Введите вопрос о Go. Команды:
  /all вопрос             показать все варианты, прошедшие порог
  /random вопрос          случайно выбрать один из прошедших порог вариантов
  /teach id | вопрос | ответ    добавить пару
  /variant id | вопрос          добавить формулировку
  /answer id | ответ            добавить вариант ответа
  /topics                 показать ID и вопросы
  /quit                   выйти
""".strip()


def dialogue(metric=AUTO, threshold=None):
    entries = load_json(PERSONAL if PERSONAL.exists() else KNOWLEDGE)
    validation = load_json(QUESTIONS)["validation"]
    engine, thresholds, reliability = fit_dialogue(entries, validation)
    print(HELP)
    while True:
        try:
            line = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        command, _, rest = line.partition(" ")
        if command == "/quit":
            return
        if not line:
            continue
        if command == "/topics":
            for entry in entries:
                print(entry["id"], "—", entry["question"])
            continue
        if command in {"/teach", "/variant", "/answer"}:
            kind, count = {"/teach": ("new", 3), "/variant": ("variant", 2), "/answer": ("answer", 2)}[command]
            fields = [field.strip() for field in rest.split("|", count - 1)]
            try:
                if len(fields) != count:
                    raise ValueError("Используйте формат команды, указанный в справке.")
                updated = learn(entries, kind, *fields)
                index = fit_dialogue(updated, validation)
                save_personal(updated)
                entries = updated
                engine, thresholds, reliability = index
                print(f"База обновлена: {PERSONAL.name}.")
            except (OSError, ValueError) as error:
                print(f"База не изменена: {error}")
            continue
        if command not in {"/all", "/random"} and command.startswith("/"):
            print(HELP)
            continue
        length, ranked = engine.rank(rest if command in {"/all", "/random"} else line)
        limits = thresholds if threshold is None else dict.fromkeys(METRICS, threshold)
        selected = adaptive_metric(length, ranked, limits, reliability) if metric == AUTO else metric
        matches = [hit for hit in ranked[selected] if hit.score >= limits[selected]]
        if not matches:
            print(NO_ANSWER)
            continue
        if command in {"/all", "/random"}:
            pool = [(hit, answer) for hit in matches for answer in engine.entries[hit.id]["answers"]]
            shown = pool if command == "/all" else [choice(pool)]
        else:
            shown = [(matches[0], engine.entries[matches[0].id]["answers"][0])]
        for hit, answer in shown:
            print(answer)
            print(f"[{hit.id}; {selected}: {hit.score:.3f}; порог {limits[selected]:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=(*METRICS, AUTO), default=AUTO)
    parser.add_argument("--threshold", type=float, help="ручной порог от 0 до 1 для всех метрик")
    args = parser.parse_args()
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold должен быть от 0 до 1")
    dialogue(args.metric, args.threshold)
