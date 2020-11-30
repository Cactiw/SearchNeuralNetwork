
from pandas_service import parse_XML
from string_service import trim_all

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def open_xml(path: str) -> pd.DataFrame:
    return parse_XML(None, ["speech", "evaluation"], "sentence", path=path)


def tokenize_row(row: pd.Series):
    return row.apply(trim_all)


def tokenize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(tokenize_row, axis=1, result_type="broadcast")  # Триммируем каждую строку
    df = df[df.evaluation.isin(frozenset({"0", "+", "-"}))]  # Удаляем некорректные оценки (не 0, +, -)

    return df


def compare_results(predicted, had_to_be):
    return np.sum(predicted == had_to_be)





train_df = open_xml("resources/news_eval_train.xml")
test_df = open_xml("resources/news_eval_test.xml")

train_df = tokenize_dataframe(train_df)
test_df = tokenize_dataframe(test_df)

train_sentences = train_df["speech"]
train_result = train_df["evaluation"]

test_sentences = test_df["speech"]
test_result = test_df["evaluation"]

vectorizer = CountVectorizer()

train_matrix = vectorizer.fit_transform(train_sentences)

vectorizer_test = CountVectorizer(vocabulary=vectorizer.vocabulary_)
test_matrix = vectorizer_test.fit_transform(test_sentences)


print(train_df)
print(test_df)


models = [MultinomialNB, SVC, LinearSVC, LogisticRegression]

for model_class in models:
    model = model_class()
    trained_model = model.fit(train_matrix, train_result)
    test_got = trained_model.predict(test_matrix)
    print(test_got)
    print(compare_results(test_got, test_result) / len(test_result) * 100)


    print("Accuracy score: " + str(accuracy_score(test_got, test_result)) + ".")
    print("F-measure: " + str(f1_score(test_got, test_result, average='macro')) + ".")

    # Можно пообщаться с моделью - раскомментировать код)
    # sent_in = None
    # while sent_in != "0":
    #     sent_in = input("Введите предложение (или 0, чтобы продолжить):")
    #     input_matrix = vectorizer_test.fit_transform([sent_in])
    #     print(trained_model.predict(input_matrix))
