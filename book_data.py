import os
import csv
import sys
import re
import pandas as pd
from surprise import Dataset
from surprise import Reader
from collections import defaultdict
import numpy as np

class BookData:

    bookID_to_name = {}
    name_to_bookID = {}
    ratingsPath = 'book-review-dataset/BX-Book-Ratings3.csv'
    booksPath = 'book-review-dataset/BX_Books.csv'

    def loadBookData(self):

        ratingsDataset = 0
        self.bookID_to_name = {}
        self.name_to_bookID = {}

        reader = Reader(line_format='user item rating', sep=';',rating_scale=(1, 10), skip_lines=1)
        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        with open(self.booksPath, newline='', encoding='ISO-8859-1') as csvfile:
                bookReader = csv.reader(csvfile, delimiter=";")
                next(bookReader)  #Skip header line
                for row in bookReader:
                    bookID = row[0]
                    bookName = row[1]
                    self.bookID_to_name[bookID] = bookName
                    self.name_to_bookID[bookName] = bookID

        return ratingsDataset

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        # change self.ratingspath
        with open(self.ratingsPath, newline='', encoding='ISO-8859-1') as csvfile:
            ratingReader = csv.reader(csvfile, delimiter=";")
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[0])
                if (user == userID):
                    # not use int
                    bookID = row[1]
                    rating = float(row[2])
                    userRatings.append((bookID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        # change self.rating
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile, delimiter=";")
            next(ratingReader)
            for row in ratingReader:
                bookID = row[1]
                ratings[bookID] += 1
        rank = 1
        for bookID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[bookID] = rank
            rank += 1
        return rankings

    def getYears(self):
        # p = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)

        with open(self.booksPath, newline='', encoding='ISO-8859-1') as csvfile:
            bookReader = csv.reader(csvfile, delimiter=";")
            next(bookReader)
            for row in bookReader:
                bookYear = row[3].replace('"', "")
                bookID = row[0]
                if bookYear:
                    years[bookID] = int(bookYear)

        return years

    def getBookName(self, bookID):
        if bookID in self.bookID_to_name:
            return self.bookID_to_name[bookID]
        else:
            return ""

    def getBookID(self, bookName):
        if bookName in self.name_to_bookID:
            return self.name_to_bookID[bookName]
        else:
            return 0
