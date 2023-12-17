# Converts hourly, monthly, and weekly salaries to yearly salaries
def convert_to_yearly(sal, pay_period):
    if pay_period == 'HOURLY':
        return sal * 2080
    elif pay_period == 'MONTHLY':
        return sal * 12
    elif pay_period == 'WEEKLY':
        return sal * 52
    else:
        return sal

# Topic modeling preprocessing
import spacy
import re
import pandas as pd

from tqdm import tqdm
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.corpora.dictionary import Dictionary

def nlp_preprocessing(texts):
    nlp = spacy.load('en_core_web_sm')
    texts = [re.sub(r'[\d?,.;:!/-]', '', x).strip().lower() for x in texts]
    texts = list(tqdm(nlp.pipe(texts, n_process=10), total=len(texts)))

    return texts

def remove_stopwords(texts):
    stop_words = list("\n firm type diploma closing date send level space job offer company firms oct connect jobs \
                    fcfa price spaces articles tax year cv job interview motivation letter advice questions mail \
                    work answers recruiters expired internship company send do activity sector have profile file \
                    company years address direction cv location benefit healthcare knowledge pet week gender status \
                    disability orientation color experience team include require skill ™ opportunity ability need ensure \
                    use care base pay position include requirement responsibility degree qualification member qualification \
                    apply application office candidate day make provide policy life applicant value part practice hour range \
                    area assign join relate change follow standard salary prefer function career regard field take age identity \
                    problem project service religion receive state tool description â€œhand".split())
    stop_words += list(STOP_WORDS)
    stop_words = set(stop_words)

    texts = [[word.lemma_ for word in doc if (word.pos_ in ['NOUN', 'VERB']) and (word.lemma_ not in stop_words)] for doc in tqdm(texts)]
    return texts


def create_words_popularity_df(texts):
    words_popularity_df = pd.DataFrame.from_records([{'id': i, 'word': word} for doc in texts for i, word in enumerate(doc)])
    words_popularity_df = words_popularity_df.drop_duplicates(subset=['id', 'word'])
    words_popularity_df = words_popularity_df.groupby(['word']).count().sort_values(by='id', ascending=False)
    words_popularity_df['doc_perc'] = words_popularity_df['id'] / len(texts)

    return words_popularity_df

def keep_n_popular_words(texts, words_popularity_df, n_words_to_keep):
    words_to_keep = set(words_popularity_df[:n_words_to_keep].index.values)
    texts = [[word for word in doc if word in words_to_keep] for doc in tqdm(texts)]
    return texts

def create_corpus(texts):
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return corpus, dictionary

def preprocess_lda_pipeline(df, n_words_to_keep=10000):
    df['Job_Desc'] = df['Job_Desc'].astype('str')
    texts = nlp_preprocessing(df.Job_Desc.values)
    texts = remove_stopwords(texts)
    words_popularity_df = create_words_popularity_df(texts)
    texts = keep_n_popular_words(texts, words_popularity_df, n_words_to_keep)
    corpus, dictionary = create_corpus(texts)
    
    return corpus, dictionary, texts