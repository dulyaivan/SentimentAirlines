import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#Вспомогательная функция получения метрик качества
def get_metrics(y_test, y_predicted):
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

#Функция для построения матрицы несоответствий (диаграмма)
def plot_confusion_matrix(cm, classes,normalize=False,
                          title='Матрица несоответствий',cmap=plt.cm.tab10):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    plt.tight_layout()
    plt.ylabel('Настоящие метки', fontsize=30)
    plt.xlabel('Предсказанные метки', fontsize=30)
    return plt

#Функция для построения диаграммы с наиболее важными словами (оказывающими влияние на тональность)
def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center',color='#1F78B4')
    plt.title('Негативные', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Влияние', fontsize=20)
    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center',color='#1F78B4')
    plt.title('Позитивные', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Влияние', fontsize=20)
    plt.subplots_adjust(wspace=0.8)
    plt.show()

#Функция для построения точечной диграммы с исходным набором данных на двух осях (PCA)
def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['#ff8c00', '#00008b']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='#ff8c00',label='Негативные')
        green_patch = mpatches.Patch(color='#00008b',label='Позитивные')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})

#Вспомогательная функция tf-idf
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train,tfidf_vectorizer

#Функиця стандартизации текста
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

#Функция для построения матрицы несоответствий
def plot_confusion_matrix(cm, classes,normalize=False,
                          title='Матрица несоответствий',cmap=plt.cm.plasma):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    plt.tight_layout()
    plt.ylabel('Настоящие метки', fontsize=30)
    plt.xlabel('Предсказанные метки', fontsize=30)
    return plt

#получение наиболее влиятельных признаков (слов)
def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}

    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes

if __name__ == '__main__':
    df = pd.read_csv('data/1600000tweets.csv',usecols=[0,5],names=['target','text'],encoding='latin-1')
    df = df.sample(frac=1).reset_index(drop=True)
    df = standardize_text(df,'text')
    list_corpus = df['text'].values
    list_labels = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.1, random_state=42)
    X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    joblib.dump(tfidf_vectorizer, 'tfidf.pkl')

    #Падает RAM при большом кол-ве точек для LSA(можно построить для уменьшенного набора данных)
    """ 
    fig = plt.figure(figsize=(16, 16))
    plot_LSA(X_train_tfidf, y_train)
    plt.show()
    """

    #Обучение модели
    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                   multi_class='multinomial', n_jobs=-1, random_state=42,verbose=True)
    clf_tfidf.fit(X_train_tfidf, y_train)
    joblib.dump(clf_tfidf, 'model.pkl')

    #Предсказание
    y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_tfidf, precision_tfidf,
                                                                           recall_tfidf, f1_tfidf))
    cm2 = confusion_matrix(y_test, y_predicted_tfidf)

    #тренировочный набор данных
    y_predicted_tfidf_train = clf_tfidf.predict(X_train_tfidf)
    cmm = confusion_matrix(y_train, y_predicted_tfidf_train)

    #Матрица несоответствий для теста
    fig = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm2, classes=['Негативные', 'Позитивные'], normalize=False,
                                 title='Матрица несоответствий')
    plt.show()
    acc = (cm2[0][0]+cm2[1][1])/(cm2[0][0]+cm2[1][1]+cm2[0][1]+cm2[1][0])
    print('acc. test = {}'.format(acc))

    # Матрица несоответствий для тренировочного набора
    fig = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cmm, classes=['Негативные', 'Позитивные'], normalize=False,
                                 title='Матрица несоответствий')
    plt.show()
    acc = (cmm[0][0] + cmm[1][1])/(cmm[0][0] + cmm[1][1] + cmm[0][1] + cmm[1][0])
    print('acc. train = {}'.format(acc))

    #Диаграмма наиболее влиятельных слов (признаков)
    importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
    top_scores = [a[0] for a in importance_tfidf[0]['tops']]
    top_words = [a[1] for a in importance_tfidf[0]['tops']]
    bottom_scores = [a[0] for a in importance_tfidf[0]['bottom']]
    bottom_words = [a[1] for a in importance_tfidf[0]['bottom']]
    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Наиболее релевантные слова")