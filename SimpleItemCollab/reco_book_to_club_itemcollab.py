
#1000000 vs 100000, 9000 vs 200000,
from book_data import BookData
from surprise import KNNBasic,KNNBaseline
import heapq
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import numpy as np
# members in clubs
testSubjects = [276822,276847,276925]

# create imaginary user with combined ratin to rep club and add to csvfile then delete
df_ratings = pd.read_csv('../book-review-dataset/BX-Book-Ratings3.csv', sep=';', encoding='ISO-8859-1')
df_members_in_club = df_ratings[df_ratings['User-ID'].isin(testSubjects)]
df_members_in_club = round(df_members_in_club.groupby('ISBN').mean()).reset_index()
df_members_in_club = df_members_in_club.reindex(columns=["User-ID","ISBN", "Book-Rating"])
df_members_in_club['Book-Rating'] = df_members_in_club['Book-Rating'].astype(np.int64)
club_rep = df_ratings['User-ID'].max() + 1
df_members_in_club['User-ID'] = club_rep
df_members_in_club.to_csv('../book-review-dataset/BX-Book-Ratings3.csv', mode='a', header=False, index=False, sep=';')
testSubject = str(club_rep)
k = 10

ml = BookData()
data = ml.loadBookData()

trainSet = data.build_full_trainset()
# 1MILLION RATINGS, 270,000 users and books vs 100,000 raitngs and 2000
sim_options = {
                'name': 'cosine',
                'user_based': False
              }

model = KNNBaseline(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()
testUserInnerID = trainSet.to_inner_uid(testSubject)

# Get the top K items we rated
testUserRatings = trainSet.ur[testUserInnerID]
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])
# kNeighbors = []
# for rating in testUserRatings:
    # if rating[1] > 9.0:
        # kNeighbors.append(rating)

# Get similar items to stuff we liked (weighted by rating)
candidates = defaultdict(float)
for itemID, rating in kNeighbors:
    similarityRow = simsMatrix[itemID]
    for innerID, score in enumerate(similarityRow):
        # CHANGE FROM 5 TO 10
        candidates[innerID] += score * (rating / 10.0)

# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        bookID = trainSet.to_raw_iid(itemID)
        #changed
        print(ml.getBookName(bookID), ratingSum)
        pos += 1
        if (pos > 10):
            break

# delete imaginary rep of club user
df_ratings = pd.read_csv('../book-review-dataset/BX-Book-Ratings3.csv', sep=';', encoding='ISO-8859-1')
df_ratings.drop(df_ratings.index[df_ratings['User-ID'] == club_rep], inplace=True)
df_ratings.to_csv('../book-review-dataset/BX-Book-Ratings3.csv',index=False, sep=';')
