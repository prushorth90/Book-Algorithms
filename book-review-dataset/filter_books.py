import os
import csv
import sys
import re
# dont need ignore
import pandas as pd
from surprise import Dataset

booksPath = '../book-review-dataset/BX_Books.csv'

books = pd.read_csv(booksPath, sep=';', encoding='ISO-8859-1')
#remove zero years
books_no_zero_year = books[books['Year-Of-Publication'] != 0]
#remove less  2022
books_less_2022 = books_no_zero_year[books_no_zero_year['Year-Of-Publication'] <2022]
#remove  url-s and url-l keep url-m
books_less_2022.drop(['Image-URL-S','Image-URL-L'], axis=1, inplace=True)
books_less_2022.to_csv('../book-review-dataset/BX-Books2.csv', sep=';',encoding='ISO-8859-1', index=False)
