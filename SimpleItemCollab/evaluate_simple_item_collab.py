from book_data import BookData
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
from surprise.model_selection import LeaveOneOut
from Evaluate.recommender_metrics import RecommenderMetrics
from Evaluate.evaluate_dataset import EvaluationData

def LoadBookData():
    ml = BookData()
    print("Loading book ratings...")
    data = ml.loadBookData()
    print("\nComputing book popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

ml, data, rankings = LoadBookData()

evalData = EvaluationData(data, rankings)

# Train on leave-One-Out train set
trainSet = evalData.GetLOOCVTrainSet()
sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

leftOutTestSet = evalData.GetLOOCVTestSet()

# Build up dict to lists of (int(bookID), predictedrating) pairs
topN = defaultdict(list)
k = 10
for uiid in range(trainSet.n_users):
    user_ratings = trainSet.ur[uiid]
    kNeighbors = heapq.nlargest(k, user_ratings, key=lambda t: t[1])

    # Get the stuff they rated, and add up ratings for each item, weighted by user similarity
    candidates = defaultdict(float)
    for itemID, rating in kNeighbors:
        similarityRow = simsMatrix[itemID]
        for innerID,score in enumerate(similarityRow):
            candidates[innerID] += score * (rating / 10.0)

    # Build a dictionary of stuff the user has already seen
    watched = {}
    for itemID, rating in trainSet.ur[uiid]:
        watched[itemID] = 1

    # Get top-rated items from similar users:
    pos = 0
    for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
        if not itemID in watched:
            bookID = trainSet.to_raw_iid(itemID)
            # change int
            topN[int(trainSet.to_raw_uid(uiid))].append( (bookID, 0.0) )
            pos += 1
            if (pos > 40):
                break

# Measure
print("HR", RecommenderMetrics.HitRate(topN, leftOutTestSet))
