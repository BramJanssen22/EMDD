import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Performance measures

def prediction_measures(y_test,y_pred):
    performance = classification_report(y_test,y_pred,output_dict=True)
    return performance['depression'],performance['normal'],performance['accuracy']



# Support measures

def percentage_depression(pred):
    return np.sum(pred == 'depression') / len(pred)

def percentage_normal(pred):
    return np.sum(pred == 'depression' / len(pred))

def true_positives(pred, true):
    return np.sum((pred == 'depression') & (true == 'depression'))

def true_negatives(pred, true):
    return np.sum((pred == 'normal') & (true == 'normal'))

def true_positive_rate(pred, true):
    return true_positives(pred,true)/(true_positives(pred,true)+true_negatives(pred,true))

def true_negative_rate(pred, true):
    return true_negatives(pred,true)/(true_positives(pred,true)+true_negatives(pred,true))

# Complete measures

def statistical_parity(predsensitive,predother):
    return percentage_depression(predsensitive)/percentage_depression(predother)

def equal_opportunity(predsensitive, truesensitive, predother, trueother):
    return true_positive_rate(predsensitive,truesensitive)/true_positive_rate(predother,trueother)

def equalised_odds(predsensitive, truesensitive, predother, trueother):
    return (true_positive_rate(predsensitive,truesensitive)+true_negative_rate(predsensitive,truesensitive))/(true_positive_rate(predother,trueother)+true_negative_rate(predsensitive,truesensitive))

def equal_accuracy(predsensitive, truesensitive, predother, trueother):
    return accuracy_score(predsensitive,truesensitive)/accuracy_score(predother,trueother)

# All measurements

def all_measures(predsensitive, truesensitive, predother, trueother, name='test',single=False):
    score_dict = {}
    score_dict['test'] = name
    depression_performance, normal_performance ,accuracy = prediction_measures(predsensitive,truesensitive)
    score_dict['depression0precision'] = depression_performance['precision']
    score_dict['depression0recall'] = depression_performance['recall']
    score_dict['depression0f1'] = depression_performance['f1-score']
    score_dict['depression0support'] = depression_performance['support']
    score_dict['normal0precision'] = normal_performance['precision']
    score_dict['normal0recall'] = normal_performance['recall']
    score_dict['normal0f1'] = normal_performance['f1-score']
    score_dict['normal0support'] = normal_performance['support']
    score_dict['accuracy0'] = accuracy
    if single == False:
        depression_performance, normal_performance ,accuracy = prediction_measures(predother,trueother)
        score_dict['depression1precision'] = depression_performance['precision']
        score_dict['depression1recall'] = depression_performance['recall']
        score_dict['depression1f1'] = depression_performance['f1-score']
        score_dict['depression1support'] = depression_performance['support']
        score_dict['normal1precision'] = normal_performance['precision']
        score_dict['normal1recall'] = normal_performance['recall']
        score_dict['normal1f1'] = normal_performance['f1-score']
        score_dict['normal1support'] = normal_performance['recall']
        score_dict['accuracy1'] = accuracy
        score_dict['statisticalParity01'] = statistical_parity(predsensitive,predother)
        score_dict['equalOpportunity01'] = equal_opportunity(predsensitive, truesensitive, predother, trueother)
        score_dict['equalisedOdds'] = equalised_odds(predsensitive, truesensitive, predother, trueother)
        score_dict['equalAccuracy'] = equal_accuracy(predsensitive, truesensitive, predother, trueother)
    return score_dict