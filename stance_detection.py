import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats, gen_or_load_feats_with_IDs
from feature_engineering import word_overlap_features, polarity_features_NLTK, keywords_features_train, keywords_features_competition
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission

from utils.system import parse_params, check_version
import lightgbm as lgb
import argparse

parser = argparse.ArgumentParser(description='Stance Detection with statistical NLP features plus classifier')
parser.add_argument('-clf', '--classifier', type=str, default='GBDT', help="GBDT or LightGBM")
# baseline features means ngram + overlapping words + sentiment by keywords while best features means ngram + overlapping words + sentiment by VADER + keywords of bodies by TF-IDF
parser.add_argument( '-f', '--feature', type=str, default='best', help='baseline or best' )
params = parser.parse_args()

def generate_features_keywords_with_IDs(stances,dataset,name, mode):
    h, b, y, IDs = [],[],[], []

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])
        IDs.append(stance['Body ID'])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features_NLTK, h, b, "features/polarity_NLTK_full."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")
    if mode == 'train':
        X_keywords = gen_or_load_feats_with_IDs(keywords_features_train, h, b,IDs, "features/keywords_."+name+".npy")
    else:
        X_keywords = gen_or_load_feats_with_IDs(keywords_features_competition, h, b,IDs, "features/keywords_."+name+".npy")
    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap, X_keywords]
    return X,y

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

if __name__ == "__main__":
    check_version()
    # parse_params()

    #Load the training dataset and generate folds
    d = DataSet()
    folds,hold_out = kfold_split(d,n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d,folds,hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    if params.feature == 'baseline':
        # Baseline features: ngram + overlapping words + sentiment by keywords
        X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
    elif params.feature == 'best':
        X_competition, y_competition = generate_features_keywords_with_IDs(competition_dataset.stances, competition_dataset, "competition", 'competition' )



    Xs = dict()
    ys = dict()

    # Load/Precompute all features now

    if params.feature == 'baseline':
        # Baseline features: ngram + overlapping words + sentiment by keywords
        X_holdout,y_holdout = generate_features(hold_out_stances,d,"holdout")
    elif params.feature == 'best':
            X_holdout,y_holdout = generate_features_keywords_with_IDs(hold_out_stances,d,"holdout", 'train')

    for fold in fold_stances:
        if params.feature == 'baseline':
        # Baseline features: ngram + overlapping words + sentiment by keywords
            Xs[fold],ys[fold] = generate_features(fold_stances[fold],d,str(fold))
        elif params.feature == 'best':
            Xs[fold],ys[fold] = generate_features_keywords_with_IDs(fold_stances[fold],d,str(fold), 'train')






    best_score = 0
    best_fold = None


    # Classifier for each fold
    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]

        if params.classifier == 'LightGBM':
            # LightGBM
            lgb_params = {'num_leaves':63, 'learning_rate': 0.1, 'num_trees':600, 'max_depth': 7, 'num_class':4, 'objective': 'multiclass'}
            num_round = 10
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test,label=y_test)
            clf = lgb.train(lgb_params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=5)
            predicted=[]
            for a in clf.predict(X_test):
                b=a.tolist()
                predicted.append(LABELS[b.index(max(b))]) 
        elif params.classifier == 'GBDT':
            clf = GradientBoostingClassifier(n_estimators=200, random_state=14128, verbose=True)
            clf.fit(X_train, y_train)
            predicted = [LABELS[int(a)] for a in clf.predict(X_test)]


        actual = [LABELS[int(a)] for a in y_test]

        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)

        score = fold_score/max_fold_score

        print("Score for fold "+ str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = clf



    #Run on Holdout set and report the final score on the holdout set
    if params.classifier == 'LightGBM':
        predicted=[]
        for a in clf.predict(X_holdout):
            b=a.tolist()
            predicted.append(LABELS[b.index(max(b))]) 
    elif params.classifier == 'GBDT':
        predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]


    actual = [LABELS[int(a)] for a in y_holdout]

    print("Scores on the dev set")
    report_score(actual,predicted)
    print("")
    print("")

    #Run on competition dataset
    if params.classifier == 'LightGBM':
        predicted=[]
        for a in clf.predict(X_competition):
            b=a.tolist()
            predicted.append(LABELS[b.index(max(b))])
    elif params.classifier == 'GBDT': 
        predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual,predicted)
