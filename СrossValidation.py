import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier

#Вспомогательная функция стандартизации текста
def  standardize_text (df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r"rt", "")
    df[text_field] = df[text_field].str.replace(r"[\U0001F600-\U0001F64F]","")
    df[text_field] = df[text_field].str.replace(r"[\U0001F300-\U0001F5FF]", "")
    df[text_field] = df[text_field].str.replace(r"[\U0001F680-\U0001F6FF]", "")
    df[text_field] = df[text_field].str.replace(r"[\U0001F1E0-\U0001F1FF]", "")
    return df

#Вспомогательная функция tf-idf
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train,tfidf_vectorizer

#Вспомогательная функция получения метрик качества
def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    df = pd.read_csv('1600000tweets.csv', usecols=[0, 5], names=['target', 'text'], encoding='latin-1')
    df = df.sample(frac=1).reset_index(drop=True)[:400000]
    df = standardize_text(df, 'text')
    list_corpus = df['text'].values
    list_labels = df['target'].values
    df = None

    kf = KFold(n_splits=5,shuffle=True)
    acc_regression=[]
    acc_gnb = []
    acc_nc = []
    acc_mlp = []
    c=0

    for train_index,test_index in kf.split(list_corpus):
        c+=1

        #split
        X_train, X_test = list_corpus[train_index], list_corpus[test_index]
        y_train, y_test = list_labels[train_index], list_labels[test_index]
        lll = len(X_train[0])

        #tfidf
        X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
        X_train = None
        X_test = None

        #Логистическая регрессия
        clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                       multi_class='multinomial', n_jobs=-1, random_state=42)
        clf_tfidf.fit(X_train_tfidf, y_train)
        y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
        accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
        acc_regression.append(accuracy_tfidf)
        print('linear_reg, acc={}'.format(accuracy_tfidf))
        clf_tfidf = None
        y_predicted_tfidf = None

        #Наивный байесовсикй классификатор
        clf_tfidf = MultinomialNB()
        clf_tfidf.fit(X_train_tfidf,y_train)
        y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
        accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
        acc_gnb.append(accuracy_tfidf)
        print('GB, acc={}'.format(accuracy_tfidf))
        clf_tfidf = None
        y_predicted_tfidf = None

        # NC
        clf_tfidf = NearestCentroid()
        clf_tfidf.fit(X_train_tfidf, y_train)
        y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
        accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
        acc_nc.append(accuracy_tfidf)
        print('NC, acc={}'.format(accuracy_tfidf))
        clf_tfidf = None
        y_predicted_tfidf = None

        #MLP
        clf_tfidf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1)
        clf_tfidf.fit(X_train_tfidf, y_train)
        y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
        accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
        acc_mlp.append(accuracy_tfidf)
        print('mlp, acc={}'.format(accuracy_tfidf))
        clf_tfidf = None
        y_predicted_tfidf = None

        print(c)
        X_train_tfidf = None
        X_test_tfidf = None

    df = pd.DataFrame({'Logistic regression':acc_regression,'GB':acc_gnb,
                       'NC':acc_nc,'MLP':acc_mlp})
    df.to_csv('cross-val_score.csv',encoding='utf-8',index=False)


