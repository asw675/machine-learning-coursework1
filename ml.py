from sklearn import tree
import numpy as np
import nltk
import sklearn
import operator
import requests
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn
import matplotlib.pyplot as plot

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import glob
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from wordcloud import WordCloud
from gensim import utils
import gensim.parsing.preprocessing as gsp
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

filters = [
    gsp.strip_tags,
    gsp.strip_punctuation,
    gsp.strip_multiple_whitespaces,
    gsp.strip_numeric,
    gsp.remove_stopwords,
    gsp.strip_short,
    gsp.stem_text
]
path = './'
business_path = path + 'bbc/business/'
entertainment_path = path + 'bbc/entertainment/'
politics_path = path + 'bbc/politics/'
sport_path = path + 'bbc/sport/'
tech_path = path + 'bbc/tech/'


def getFileName(name):
    if name < 10:
        return '00' + str(name) + '.txt'
    elif name >= 10 and name < 100:
        return '0' + str(name) + '.txt'
    else:
        return str(name) + '.txt'


business_data = []
entertainment_data = []
politics_data = []
sport_data = []
tech_data = []

business_data_dev = []
entertainment_data_dev = []
politics_data_dev = []
sport_data_dev = []
tech_data_dev = []

business_data_test = []
entertainment_data_test = []
politics_data_test = []
sport_data_test = []
tech_data_test = []

readdict = {0: business_path, 1: entertainment_path, 2: politics_path, 3: sport_path, 4: tech_path}
writedict = {0: business_data, 1: entertainment_data, 2: politics_data, 3: sport_data, 4: tech_data}
writedictdev = {0: business_data_dev, 1: entertainment_data_dev, 2: politics_data_dev, 3: sport_data_dev,
                4: tech_data_dev}
writedicttest = {0: business_data_test, 1: entertainment_data_test, 2: politics_data_test, 3: sport_data_test,
                 4: tech_data_test}
predict_models = []


def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s


for k in range(5):
    length = len(glob.glob(readdict[k] + '*.txt'))
    length *= 0.8
    length = int(length)
    for i in range(1, length):
        realPath = readdict[k] + getFileName(i)
        with open(realPath, 'r', encoding="ISO-8859-1") as f:
            for line in f.readlines():
                line = line.strip('\n')
                if (len(line) > 0):
                    writedict[k].append(clean_text(line))

for k in range(5):
    length = len(glob.glob(readdict[k] + '*.txt'))
    length8 = length * 0.8
    length9 = length * 0.9
    length8 = int(length8)
    length9 = int(length9)
    for i in range(length8, length9):
        realPath = readdict[k] + getFileName(i)
        with open(realPath, 'r', encoding="ISO-8859-1") as f:
            for line in f.readlines():
                line = line.strip('\n')
                if (len(line) > 0):
                    writedictdev[k].append(clean_text(line))
    for i in range(length9, length):
        realPath = readdict[k] + getFileName(i)
        with open(realPath, 'r', encoding="ISO-8859-1") as f:
            for line in f.readlines():
                line = line.strip('\n')
                if (len(line) > 0):
                    writedicttest[k].append(clean_text(line))

lemmatizer = nltk.stem.WordNetLemmatizer()


def get_list_tokens(string):
    sentence_split = nltk.tokenize.sent_tokenize(string)
    list_tokens = []
    for sentence in sentence_split:
        list_tokens_sentence = nltk.tokenize.word_tokenize(sentence)
        for token in list_tokens_sentence:
            list_tokens.append(lemmatizer.lemmatize(token).lower())
    return list_tokens


def get_target_words(string):
    text = get_list_tokens(string)
    list_tokens = []
    for words in nltk.pos_tag(text):
        if words[1] == "VBP" or words[1] == "VB" or words[1] == "NN":
            list_tokens.append(words[0])
    return list_tokens


stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")

dict_word_frequency = {}
for k in range(5):
    for review in writedict[k]:
        sentence_tokens = get_list_tokens(review)
        for word in sentence_tokens:
            if word in stopwords: continue
            if word not in dict_word_frequency:
                dict_word_frequency[word] = 1
            else:
                dict_word_frequency[word] += 1

dict_word_frequency_only = {}
for k in range(5):
    for review in writedict[k]:
        sentence_tokens = get_target_words(review)
        for word in sentence_tokens:
            if word in stopwords: continue
            if word not in dict_word_frequency_only:
                dict_word_frequency_only[word] = 1
            else:
                dict_word_frequency_only[word] += 1


def get_vector_text(list_vocab, string):
    vector_text = np.zeros(len(list_vocab))
    list_tokens_string = get_list_tokens(string)
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i] = list_tokens_string.count(word)
    return vector_text


def generate_N_grams(text, ngram=1):
    ans = []
    tokens = get_list_tokens(clean_text(text))
    for i in range(len(tokens) - ngram + 1):
        gram = ""
        for j in range(ngram):
            gram += (tokens[i + j])
            if j < ngram - 1:
                gram += (" ")
        ans.append(gram)
    return ans


def get_vector_gramtext(list_vocab, string):
    vector_text = np.zeros(len(list_vocab))
    list_tokens_string = []
    list_tokens_string.append(generate_N_grams(string, 1))
    list_tokens_string.extend(generate_N_grams(string, 2))
    for i, word in enumerate(list_vocab):
        if word in list_tokens_string:
            vector_text[i] = list_tokens_string.count(word)
    return vector_text


list_num_features = [250, 500, 750, 1000]
best_accuracy_dev = [0.0, 0.0, 0.0, 0.0, 0.0]
best_features = [0, 0, 0, 0, 0]
Y_test_labels = []
Y_predict_labels = []
combine_name = ['frequency', 'frequency + TFIDF', 'frequency + part of speech', 'frequency + part of speech + TFIDF',
                'frequency + ngram']
for i in range(4):
    sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:list_num_features[i]]
    sorted_list_only = sorted(dict_word_frequency_only.items(), key=operator.itemgetter(1), reverse=True)[
                       :list_num_features[i]]

    vocabulary = []
    for word, frequency in sorted_list:
        vocabulary.append(word)

    vocabulary_only = []
    for word, frequency in sorted_list_only:
        vocabulary_only.append(word)

    X_train = []
    Y_train = []
    for k in range(5):
        for review in writedict[k]:
            vector_review = get_vector_text(vocabulary, review)
            X_train.append(vector_review)
            Y_train.append(k)
    X_dev = []
    Y_dev = []
    for k in range(5):
        for review in writedictdev[k]:
            vector_review = get_vector_text(vocabulary, review)
            X_dev.append(vector_review)
            Y_dev.append(k)

    X_train_sentanalysis = np.asarray(X_train)
    Y_train_sentanalysis = np.asarray(Y_train)
    X_dev_sentanalysis = np.asarray(X_dev)
    Y_dev_sentanalysis = np.asarray(Y_dev)
    TFIDF_X_train = TfidfTransformer().fit_transform(X_train_sentanalysis)
    TFIDF_X_dev = TfidfTransformer().fit_transform(X_dev_sentanalysis)

    svm_clf_sentanalysis_ = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf_sentanalysis_.fit(X_train_sentanalysis, Y_train_sentanalysis)
    svm_clf_sentanalysis_TFIDF = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf_sentanalysis_TFIDF.fit(TFIDF_X_train, Y_train_sentanalysis)

    X_train_only = []
    Y_train_only = []
    for k in range(5):
        for review in writedict[k]:
            vector_review = get_vector_text(vocabulary_only, review)
            X_train_only.append(vector_review)
            Y_train_only.append(k)
    X_dev_only = []
    Y_dev_only = []
    for k in range(5):
        for review in writedictdev[k]:
            vector_review = get_vector_text(vocabulary_only, review)
            X_dev_only.append(vector_review)
            Y_dev_only.append(k)

    X_train_sentanalysis_only = np.asarray(X_train_only)
    Y_train_sentanalysis_only = np.asarray(Y_train_only)
    X_dev_sentanalysis_only = np.asarray(X_dev_only)
    Y_dev_sentanalysis_only = np.asarray(Y_dev_only)

    TFIDF_X_train_only = TfidfTransformer().fit_transform(X_train_sentanalysis_only)
    TFIDF_X_dev_only = TfidfTransformer().fit_transform(X_dev_sentanalysis_only)

    svm_clf_sentanalysis_only = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf_sentanalysis_only.fit(X_train_sentanalysis_only, Y_train_sentanalysis_only)
    svm_clf_sentanalysis_TFIDF_only = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf_sentanalysis_TFIDF_only.fit(TFIDF_X_train_only, Y_train_sentanalysis_only)

    dict_word_gram = {}
    for k in range(5):
        for review in writedict[k]:
            for i in range(1, 2):
                for word in generate_N_grams(review, i + 1):
                    if word not in dict_word_gram:
                        dict_word_gram[word] = 1
                    else:
                        dict_word_gram[word] += 1

    gram_sorted_list = sorted(dict_word_gram.items(), key=operator.itemgetter(1), reverse=True)[:list_num_features[i]]

    gram_vocabulary = []
    for word, frequency in gram_sorted_list:
        gram_vocabulary.append(word)
    gram_vocabulary.extend(vocabulary)

    X_train_ngram = []
    Y_train_ngram = []
    for k in range(5):
        for review in writedict[k]:
            vector_review = get_vector_gramtext(gram_vocabulary, review)
            X_train_ngram.append(vector_review)
            Y_train_ngram.append(k)
    X_dev_ngram = []
    Y_dev_ngram = []
    for k in range(5):
        for review in writedictdev[k]:
            vector_review = get_vector_text(gram_vocabulary, review)
            X_dev_ngram.append(vector_review)
            Y_dev_ngram.append(k)

    X_train_sentanalysis_ngram = np.asarray(X_train_ngram)
    Y_train_sentanalysis_ngram = np.asarray(Y_train_ngram)
    X_dev_sentanalysis_ngram = np.asarray(X_dev_ngram)
    Y_dev_sentanalysis_ngram = np.asarray(Y_dev_ngram)

    svm_clf_sentanalysis_gram = sklearn.svm.SVC(kernel="linear", gamma='auto')
    svm_clf_sentanalysis_gram.fit(X_train_sentanalysis_ngram, Y_train_sentanalysis_ngram)
    svm_clf_sentanalysis_TFIDF_gram = sklearn.svm.SVC(kernel="linear", gamma='auto')

    predict_normal = svm_clf_sentanalysis_.predict(X_dev_sentanalysis)
    TFIDF_predict = svm_clf_sentanalysis_TFIDF.predict(TFIDF_X_dev)
    predict_only = svm_clf_sentanalysis_only.predict(X_dev_sentanalysis_only)
    TFIDF_predict_only = svm_clf_sentanalysis_TFIDF_only.predict(TFIDF_X_dev_only)
    predict_ngram = svm_clf_sentanalysis_gram.predict(X_dev_ngram)

    accuracy = accuracy_score(Y_dev_sentanalysis, predict_normal)
    TFIDF_accuracy = accuracy_score(Y_dev_sentanalysis, TFIDF_predict)

    accuracy_only = accuracy_score(Y_dev_sentanalysis, predict_only)
    TFIDF_accuracy_only = accuracy_score(Y_dev_sentanalysis, TFIDF_predict_only)

    accuracy_ngram = accuracy_score(Y_dev_sentanalysis_ngram, predict_ngram)

    if accuracy > best_accuracy_dev[0]:
        best_accuracy_dev[0] = accuracy
        best_features[0] = list_num_features[i]

    if TFIDF_accuracy > best_accuracy_dev[1]:
        best_accuracy_dev[1] = TFIDF_accuracy
        best_features[1] = list_num_features[i]

    if accuracy_only > best_accuracy_dev[2]:
        best_accuracy_dev[2] = accuracy_only
        best_features[2] = list_num_features[i]

    if TFIDF_accuracy_only > best_accuracy_dev[3]:
        best_accuracy_dev[3] = TFIDF_accuracy_only
        best_features[3] = list_num_features[i]

    if accuracy_ngram > best_accuracy_dev[4]:
        best_accuracy_dev[4] = accuracy_ngram
        best_features[4] = list_num_features[i]

sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:best_features[0]]
sorted_list_only = sorted(dict_word_frequency_only.items(), key=operator.itemgetter(1), reverse=True)[:best_features[2]]

vocabulary = []
for word, frequency in sorted_list:
    vocabulary.append(word)

vocabulary_only = []
for word, frequency in sorted_list_only:
    vocabulary_only.append(word)

X_train = []
Y_train = []
for k in range(5):
    for review in writedict[k]:
        vector_review = get_vector_text(vocabulary, review)
        X_train.append(vector_review)
        Y_train.append(k)
X_test = []
Y_test = []
for k in range(5):
    for review in writedicttest[k]:
        vector_review = get_vector_text(vocabulary, review)
        X_test.append(vector_review)
        Y_test.append(k)

X_train_sentanalysis = np.asarray(X_train)
Y_train_sentanalysis = np.asarray(Y_train)
X_test_sentanalysis = np.asarray(X_test)
Y_test_sentanalysis = np.asarray(Y_test)

TFIDF_X_train = TfidfTransformer().fit_transform(X_train_sentanalysis)
TFIDF_X_test = TfidfTransformer().fit_transform(X_test_sentanalysis)

svm_clf_sentanalysis_ = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_sentanalysis_.fit(X_train_sentanalysis, Y_train_sentanalysis)
svm_clf_sentanalysis_TFIDF = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_sentanalysis_TFIDF.fit(TFIDF_X_train, Y_train_sentanalysis)

X_train_only = []
Y_train_only = []
for k in range(5):
    for review in writedict[k]:
        vector_review = get_vector_text(vocabulary_only, review)
        X_train_only.append(vector_review)
        Y_train_only.append(k)
X_test_only = []
Y_test_only = []
for k in range(5):
    for review in writedicttest[k]:
        vector_review = get_vector_text(vocabulary_only, review)
        X_test_only.append(vector_review)
        Y_test_only.append(k)

X_train_sentanalysis_only = np.asarray(X_train_only)
Y_train_sentanalysis_only = np.asarray(Y_train_only)
X_test_sentanalysis_only = np.asarray(X_test_only)
Y_test_sentanalysis_only = np.asarray(Y_test_only)

TFIDF_X_train_only = TfidfTransformer().fit_transform(X_train_sentanalysis_only)
TFIDF_X_test_only = TfidfTransformer().fit_transform(X_test_sentanalysis_only)

svm_clf_sentanalysis_only = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_sentanalysis_only.fit(X_train_sentanalysis_only, Y_train_sentanalysis_only)
svm_clf_sentanalysis_TFIDF_only = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_sentanalysis_TFIDF_only.fit(TFIDF_X_train_only, Y_train_sentanalysis_only)

dict_word_gram = {}
for k in range(5):
    for review in writedict[k]:
        for i in range(1, 2):
            for word in generate_N_grams(review, i + 1):
                if word not in dict_word_gram:
                    dict_word_gram[word] = 1
                else:
                    dict_word_gram[word] += 1

gram_sorted_list = sorted(dict_word_gram.items(), key=operator.itemgetter(1), reverse=True)[:list_num_features[i]]

gram_vocabulary = []
for word, frequency in gram_sorted_list:
    gram_vocabulary.append(word)
gram_vocabulary.extend(vocabulary)

X_train_ngram = []
Y_train_ngram = []
for k in range(5):
    for review in writedict[k]:
        vector_review = get_vector_gramtext(gram_vocabulary, review)
        X_train_ngram.append(vector_review)
        Y_train_ngram.append(k)

X_test_ngram = []
Y_test_ngram = []
for k in range(5):
    for review in writedicttest[k]:
        vector_review = get_vector_text(gram_vocabulary, review)
        X_test_ngram.append(vector_review)
        Y_test_ngram.append(k)

X_train_sentanalysis_ngram = np.asarray(X_train_ngram)
Y_train_sentanalysis_ngram = np.asarray(Y_train_ngram)
X_test_sentanalysis_ngram = np.asarray(X_test_ngram)
Y_test_sentanalysis_ngram = np.asarray(Y_test_ngram)

svm_clf_sentanalysis_gram = sklearn.svm.SVC(kernel="linear", gamma='auto')
svm_clf_sentanalysis_gram.fit(X_train_sentanalysis_ngram, Y_train_sentanalysis_ngram)
svm_clf_sentanalysis_TFIDF_gram = sklearn.svm.SVC(kernel="linear", gamma='auto')

predict_normal = svm_clf_sentanalysis_.predict(X_test_sentanalysis)
TFIDF_predict = svm_clf_sentanalysis_TFIDF.predict(TFIDF_X_test)
predict_only = svm_clf_sentanalysis_only.predict(X_test_sentanalysis_only)
TFIDF_predict_only = svm_clf_sentanalysis_TFIDF_only.predict(TFIDF_X_test_only)
predict_ngram = svm_clf_sentanalysis_gram.predict(X_test_ngram)

Y_test_labels.append(Y_test_sentanalysis)
Y_test_labels.append(Y_test_sentanalysis)
Y_test_labels.append(Y_test_sentanalysis_only)
Y_test_labels.append(Y_test_sentanalysis_only)
Y_test_labels.append(Y_test_sentanalysis_ngram)
Y_predict_labels.append(predict_normal)
Y_predict_labels.append(TFIDF_predict)
Y_predict_labels.append(predict_only)
Y_predict_labels.append(TFIDF_predict_only)
Y_predict_labels.append(predict_ngram)
for i in range(5):
    precision = precision_score(Y_test_labels[i], Y_predict_labels[i], average='macro')
    recall = recall_score(Y_test_labels[i], Y_predict_labels[i], average='macro')
    f1 = f1_score(Y_test_labels[i], Y_predict_labels[i], average='macro')
    accuracy = accuracy_score(Y_test_labels[i], Y_predict_labels[i])

    print(combine_name[i] + "  accuracy:  " + str(accuracy))
    print(combine_name[i] + "  precision:  " + str(precision))
    print(combine_name[i] + "  recall:  " + str(recall))
    print(combine_name[i] + "  f1:  " + str(f1))

