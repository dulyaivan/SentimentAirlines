import pandas as pd
import collections
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

#Вспомогательная функция для стандартизации текста
def  standardize_text (df, text_field):
    df['clean'] = df[text_field].str.replace(r"http\S+", "")
    df['clean'] = df[text_field].str.replace(r"http", "")
    df['clean'] = df[text_field].str.replace(r"@\S+", "")
    df['clean'] = df[text_field].str.replace(r"@", "at")
    df['clean'] = df[text_field].str.lower()
    df['clean'] = df[text_field].str.replace(r"rt", "")
    df['clean'] = df[text_field].str.replace(r"[\U0001F600-\U0001F64F]","")
    df['clean'] = df[text_field].str.replace(r"[\U0001F300-\U0001F5FF]", "")
    df['clean'] = df[text_field].str.replace(r"[\U0001F680-\U0001F6FF]", "")
    df['clean'] = df[text_field].str.replace(r"[\U0001F1E0-\U0001F1FF]", "")
    return df

if __name__ == '__main__':
    #загрука корпуса стоп-слов
    nltk.download('stopwords')
    stopWords = set(stopwords.words('english'))

    #Импорт данных
    df = pd.read_csv('data/Airlines.csv')
    df = standardize_text(df, 'text')
    df = df.loc[df['target'] == 'american airlines']

    #Получение N-грамм для позитивных и негативных публикаций
    for semantic in ['positive','negative']:
        current_df = df.loc[df['prediction'] == semantic]
        text = current_df['clean'].values
        allbigrams = []
        allunigrams = []
        alltrigrams = []
        for line in text:
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(line)
            tokens = [word for word in tokens if not word in stopWords]
            bigrams = list(nltk.ngrams(tokens, 2))
            trigrams = list(nltk.ngrams(tokens, 3))
            for i in tokens:
                allunigrams.append(i)
            for i in bigrams:
                allbigrams.append(i)
            for i in trigrams:
                alltrigrams.append(i)
        unigrams = collections.Counter(allunigrams).most_common(100)
        bigrams = collections.Counter(allbigrams).most_common(100)
        trigrams = collections.Counter(alltrigrams).most_common(100)
        unigrams = pd.DataFrame({'ngram':[n_gram for n_gram, c in unigrams],
        'count': [c for n_gram, c in unigrams]})
        bigrams = pd.DataFrame({'ngram':[n_gram for n_gram, c in bigrams],
        'count': [c for n_gram, c in bigrams]})
        trigrams = pd.DataFrame({'ngram': [n_gram for n_gram, c in trigrams],
                                'count': [c for n_gram, c in trigrams]})
        unigrams.to_csv('American_uni_{}'.format(semantic))
        bigrams.to_csv('American_bi_{}'.format(semantic))
        trigrams.to_csv('American_tri_{}'.format(semantic))