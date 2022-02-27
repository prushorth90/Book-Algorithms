from surprise import AlgoBase
from surprise import PredictionImpossible
from book_data import BookData
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        # Compute item similarity matrix based on content attributes

        ml = BookData()
        years = ml.getYears()

        print("Computing content-based similarity matrix...")

        # Compute distance for every book combination as a 2x2 matrix
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))

        for thisRating in range(self.trainset.n_items):
            if (thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                # thisbookID = int(self.trainset.to_raw_iid(thisRating))
                thisBookID = self.trainset.to_raw_iid(thisRating)
                otherBookID = self.trainset.to_raw_iid(otherRating)
                yearSimilarity = self.computeYearSimilarity(thisBookID, otherBookID, years)
                self.similarities[thisRating, otherRating] = yearSimilarity
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]

        print("...done.")

        return self

    def computeYearSimilarity(self, book1, book2, years):
        diff = abs(years[book1] - years[book2])
        sim = math.exp(-diff / 10.0)
        return sim

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        for rating in self.trainset.ur[u]:
            yearSimilarity = self.similarities[i,rating[0]]
            neighbors.append( (yearSimilarity, rating[1]) )

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating

        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
