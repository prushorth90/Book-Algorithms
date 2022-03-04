import os
import csv
import sys
import re
# dont need ignore
import pandas as pd
from surprise import Dataset

ratingsPath = '../book-review-dataset/BX-Book-Ratings2.csv'
booksPath = '../book-review-dataset/BX-Books2.csv'

ratings = pd.read_csv(ratingsPath, sep=';', encoding='ISO-8859-1')
ratings_no_zero = ratings[ratings['Book-Rating'] != 0]
# print(len(df_ratings_explicit))

# get rid of invalid isbn eg funnysauce
books = pd.read_csv(booksPath, sep=';', encoding='ISO-8859-1')
isbn = books['ISBN'].to_list()
ratings_no_zero = ratings_no_zero[ratings_no_zero['ISBN'].isin(isbn)]

min_books_user_rated = 15
num_book_user_rated = ratings_no_zero['User-ID'].value_counts()
users = num_book_user_rated[num_book_user_rated >= min_books_user_rated].index.to_list()
user_match = ratings_no_zero[ratings_no_zero['User-ID'].isin(users)]

min_book_rated = 10
num_book_rated = user_match['ISBN'].value_counts()
books = num_book_rated[num_book_rated >= min_book_rated].index.to_list()
book_match = user_match[user_match['ISBN'].isin(books)]

book_match.to_csv('../book-review-dataset/BX-Book-Ratings3.csv', sep=';',encoding='ISO-8859-1', index=False)

# END =----------------------------
