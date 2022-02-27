from book_data import BookData
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

testSubject = '277427'
k = 10

# Load our data set and compute the user similarity matrix
ml = BookData()
data = ml.loadBookData()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID]
# print(similarityRow)

similarUsers = []
for innerID, score in enumerate(similarityRow):
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )
# print(similarUsers)
kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])
# kNeighbors = []
# for rating in similarUsers:
#     if rating[1] > 0.95:
#         kNeighbors.append(rating)


# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID]
    for rating in theirRatings:
        #change 5 to 10
        candidates[rating[0]] += (rating[1] / 10) * userSimilarityScore

# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1

# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        bookID = trainSet.to_raw_iid(itemID)
        #CHANNGED
        print(ml.getBookName(bookID), ratingSum)
        pos += 1
        if (pos > 10):
            break
