from Evaluate.evaluate_dataset_fast import EvaluationData
from book_data import BookData
#CHANGED
from Evaluate.evaluate_algorithm import EvaluatedAlgorithm
from surprise import SVD, SVDpp



import pandas as pd
import numpy as np

# members in clubs
testSubjects = [276822,276847,276925]

df_ratings = pd.read_csv('../book-review-dataset/BX-Book-Ratings3.csv', sep=';', encoding='ISO-8859-1')
df_members_in_club = df_ratings[df_ratings['User-ID'].isin(testSubjects)]
df_members_in_club = round(df_members_in_club.groupby('ISBN').mean()).reset_index()
df_members_in_club = df_members_in_club.reindex(columns=["User-ID","ISBN", "Book-Rating"])
df_members_in_club['Book-Rating'] = df_members_in_club['Book-Rating'].astype(np.int64)
club_rep = df_ratings['User-ID'].max() + 1
df_members_in_club['User-ID'] = club_rep
df_members_in_club.to_csv('../book-review-dataset/BX-Book-Ratings3.csv', mode='a', header=False, index=False, sep=';')
testSubject = club_rep


def SampleTopNRecs(dataset,ml, testSubject, k=10):
    # print("\nUsing recommender ", algo.GetName())

    algo = EvaluatedAlgorithm(SVD(), "SVD")


    print("\nBuilding recommendation model...")
    trainSet = dataset.GetFullTrainSet()
    algo.GetAlgorithm().fit(trainSet)

    print("Computing recommendations...")
    testSet = dataset.GetAntiTestSetForUser(testSubject)

    predictions = algo.GetAlgorithm().test(testSet)

    recommendations = []

    print ("\nWe recommend:")
    for userID, bookID, actualRating, estimatedRating, _ in predictions:
        strBookID = bookID
        recommendations.append((strBookID, estimatedRating))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    for ratings in recommendations[:10]:
        print(ml.getBookName(ratings[0]), ratings[1])



def LoadBookData():
    ml = BookData()
    print("Loading book ratings...")
    data = ml.loadBookData()
    print("\nComputing book popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)


#delete club rep
(ml, evaluationData, rankings) = LoadBookData()
ed = EvaluationData(evaluationData, rankings)
SampleTopNRecs(dataset=ed, ml=ml, testSubject=testSubject)
df_ratings = pd.read_csv('../book-review-dataset/BX-Book-Ratings3.csv', sep=';', encoding='ISO-8859-1')
df_ratings.drop(df_ratings.index[df_ratings['User-ID'] == club_rep], inplace=True)
# df_ratings['Book-Rating'] = df_ratings['Book-Rating'].astype(np.int64)
df_ratings.to_csv('../book-review-dataset/BX-Book-Ratings3.csv',index=False, sep=';')
