
import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
# from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import os
import numpy as np
import joblib
from joblib import dump, load

def preprocessData(reviewtext):
    sw = set(stopwords.words('english'))
    reviewtext = re.sub(r'[^\w\s]', '', reviewtext)
    reviewtext = re.sub(r'\d+', '', reviewtext)
    reviewtext = reviewtext.lower()
    splitData = reviewtext.split()
    stemmedData = [PorterStemmer().stem(x) for x in splitData if x not in sw]
    return ' '.join(stemmedData)

def priorProbability(ratings):
    classes, total = np.unique(ratings, return_counts=True)
    return {cl: tot / len(ratings) for cl, tot in zip(classes, total)}

def likelihoodProbability(trainFeatures, ratings):
    classes = np.unique(ratings)
    likelihoodProb = {}
    for i in classes:
        classIndex = np.where(ratings == i)[0]
        classSamples = trainFeatures[classIndex, :]
        classFeaturecounts = np.sum(classSamples, axis=0) + 1
        classProb = classFeaturecounts / (np.sum(classFeaturecounts))
        likelihoodProb[i] = classProb
    return likelihoodProb

def predictRating(trainFeatures, priorProb, likelihoodProb):
    samples, features = trainFeatures.shape
    classes = list(priorProb.keys())
    n = 5
    predictions = np.zeros(samples, dtype=int)
    for i in range(samples):
        featuresMatrix = trainFeatures[i, :]
        posteriorProb = np.zeros(n)
        for j in range(n):
            clas = classes[j]
            priorprob= priorProb[clas]
            likelihoodprob = np.sum(featuresMatrix * np.log(likelihoodProb[clas]) + (1 - featuresMatrix) * np.log(1 - likelihoodProb[clas]))
            posteriorProb[j] = np.log(priorprob) + likelihoodprob
        predictions[i] = classes[np.argmax(posteriorProb)]
    return predictions


def vectorize(train_data, val_data, maxFeatures, min):
    vectorizer = CountVectorizer(max_features=maxFeatures, min_df=min)
    train_text = train_data['text'].apply(preprocessData)
    val_text = val_data['text'].apply(preprocessData)
    train_features= vectorizer.fit_transform(train_text)
    val_features = vectorizer.transform(val_text)

    return train_text, train_features, val_text, val_features

def naiveBayes():

    totalData = []
    count=0
    with open('yelp_academic_dataset_review.json', 'r', encoding='utf-8') as f:
        for x in f:
            totalData.append(json.loads(x))
            count += 1
            if count == 100000:  
                break
    df = pd.DataFrame(totalData)
    # print(df.shape)

    df['useful'] = df['useful'].clip(upper=5)
    df['funny'] = df['funny'].clip(upper=5)
    df['cool'] = df['cool'].clip(upper=5)

    df['useful'] = df['useful'].clip(lower=1)
    df['funny'] = df['funny'].clip(lower=1)
    df['cool'] = df['cool'].clip(lower=1)

    train_data, test_data = train_test_split(df, test_size=0.10, random_state=0)
    train_data, val_data = train_test_split(train_data, test_size=0.10, random_state=0)

    # test_data.to_json('test.jsonl', orient='records', lines=True)

    train_data.dropna(subset=['text', 'stars', 'useful', 'funny', 'cool'], inplace=True)
    maxFeatures = 5000
    min = 10

    train_text, train_features, val_text, val_features= vectorize(train_data, val_data, maxFeatures, min)
    train_features = train_features.astype('int8')
    val_features=val_features.astype('int8')

    priorProb_stars = priorProbability(train_data['stars'])
    print('\nStars\n')
    Prob_likelihood = likelihoodProbability(train_features.toarray(), train_data['stars'])
    val_prediction_stars = predictRating(val_features.toarray(), priorProb_stars, Prob_likelihood)
    macroF1 = f1_score(val_data['stars'], val_prediction_stars, average='macro')
    print("Macro F1 score: {:.4f}".format(macroF1))
    print('\n')
    print(classification_report(val_data['stars'], val_prediction_stars))

    joblib.dump((train_features, priorProb_stars, Prob_likelihood), 'naive_bayes_model_stars.joblib')
    

    print('Useful\n')
    priorProb_useful = priorProbability(train_data['useful'])
    prob_likelihood_useful = likelihoodProbability(train_features.toarray(), train_data['useful'])
    val_prediction_useful = predictRating(val_features.toarray(), priorProb_useful, prob_likelihood_useful)
    macroF1_useful = f1_score(val_data['useful'], val_prediction_useful, average='macro')
    print("Useful - Macro F1 score: {:.4f}".format(macroF1_useful))
    print('\n')
    print(classification_report(val_data['useful'], val_prediction_useful))

    joblib.dump((train_features, priorProb_useful, prob_likelihood_useful), 'naive_bayes_model_useful.joblib')

    print('Funny\n')
    priorProb_funny = priorProbability(train_data['funny'])
    prob_likelihood_funny = likelihoodProbability(train_features.toarray(), train_data['funny'])
    val_prediction_funny = predictRating(val_features.toarray(), priorProb_funny, prob_likelihood_funny)
    macroF1_funny = f1_score(val_data['funny'], val_prediction_funny, average='macro')
    print("Funny - Macro F1 score: {:.4f}".format(macroF1_funny))
    print('\n')
    print(classification_report(val_data['funny'], val_prediction_funny))

    joblib.dump((train_features, priorProb_funny, prob_likelihood_funny), 'naive_bayes_model_funny.joblib')

    print('Cool\n')
    priorProb_cool = priorProbability(train_data['cool'])
    prob_likelihood_cool = likelihoodProbability(train_features.toarray(), train_data['cool'])
    val_prediction_cool = predictRating(val_features.toarray(), priorProb_cool, prob_likelihood_cool)
    macroF1_cool = f1_score(val_data['cool'], val_prediction_cool, average='macro')
    print("Cool - Macro F1 score: {:.4f}".format(macroF1_cool))
    print('\n')
    print(classification_report(val_data['cool'], val_prediction_cool))

    joblib.dump((train_features, priorProb_cool, prob_likelihood_cool), 'naive_bayes_model_cool.joblib')

naiveBayes()

