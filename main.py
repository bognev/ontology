import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from collections import Counter
from pymorphy2 import MorphAnalyzer
import os


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@]')
BAD_SYMBOLS_RE = re.compile('[^0-9а-я #+_]')
STOPWORDS = set(stopwords.words('russian'))

def normalization(text):
    words = []
    text = text.lower()  # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BeautifulSoup(text, features='lxml').get_text()
    text = nltk.word_tokenize(text)
    morph = MorphAnalyzer(lang='ru')
    for part in text:
        part = re.sub(BAD_SYMBOLS_RE, "", part)  # delete symbols which are in BAD_SYMBOLS_RE from text
        part = " ".join([word for word in part.split() if not word in STOPWORDS])  # delete stopwords from text
        if not part in '' and not part.isdigit():
            parse = morph.parse(part)
            tmp = parse[0].inflect({'plur', 'nomn'})
            if tmp is not None:
                part = tmp.word
                words.append(part.capitalize())

    words = " ".join([word for word in words])# if not word in ''])
    # words = stem.lemmatize(words)

    return words

def freq_dist(data):
    ngram_vectorizer = CountVectorizer(analyzer='word', tokenizer=word_tokenize, ngram_range=(1,1), min_df=1)
    X = ngram_vectorizer.fit_transform(data.split())
    vocab = list(ngram_vectorizer.get_feature_names())
    counts = X.sum(axis=0).A1
    return Counter(dict(zip(vocab, counts)))

def inOrder(root, category):
    categories_info = []
    stack = []
    for node in root:
        stack.append(node)
    # visited = {}
    height = {}
    level = -1
    log = False
    while stack:
        top = stack.pop()
        # visited[top['id']] = True
        height[top['id']] = 1 + height.get(top['category'], -1)
        if height[top['id']] <= level:
            log = False
        if 'children' in top:
            for child in top['children']:
                # if not visited.get(child['id'], False):
                stack.append(child)

        if top['id'] in category:
            level = height[top['id']]
            super_category = top['id']
            log = True
        if log:
            # dct = {'id': top['id'],'term' : top['term'], 'category': top['category']}
            dct = {'id': top['id'], 'term': top['term'], 'category': super_category}
            categories_info.append(dct)
    # print(height)

    # print(res)
    return categories_info


def build_ontology(ontology_path, categories):
    df = pd.read_csv(ontology_path, sep='\t', names=['id', 'category', 'parent_id'])
    df['parent_id'].fillna(df['id'], inplace=True)
    df['parent_id'] = df['parent_id'].astype('int')

    nodes = {}
    for index, row in df.iterrows():
        nodes[row.id] = {'id': row.id, 'term': row.category, 'category': row.parent_id}

    tree = []
    for index, row in df.iterrows():
        id, parent_id = row.id, row.parent_id
        node = nodes[id]

        # either make the node a new tree or link it to its parent
        if id == parent_id:
            # start a new tree in the tree
            tree.append(node)
        else:
            # add new_node as child to parent
            parent = nodes[parent_id]
            if not 'children' in parent:
                # ensure parent has a 'children' field
                parent['children'] = []
            children = parent['children']
            children.append(node)

    # for category in categories:
    categories_info = inOrder(tree, categories)
    return df, categories_info

def calculate_stat (text_paths, categories_info):
    stat = []
    # for text_path in text_paths:
    #     head, file_name = os.path.split(text_path)
    #     dct = {
    #         'file_name': file_name,
    #         'term_categories': {}
    #         }
    #     stat.append(dct)

    for i, text_path in enumerate(text_paths):
        head, file_name = os.path.split(text_path)
        corpus = open(text_path, encoding="utf-8").read()
        corpus = normalization(corpus)
        # print(corpus)
        # print(categories_info)
        frequency = freq_dist(corpus)
        # print(frequency)
        full_dct = {
            'file_name': file_name,
            'term_categories': {}
        }
        # stat = []
        super_category = {}

        for category in categories_info:
            if i == 0:
                print(category)
            found_terms = []
            total_terms = 0
            key = category['term'].lower()
            nums = frequency[key]
            if nums:
                if super_category.get(category['category']) is None:
                    found_terms.append(category['term'])
                    total_terms += nums
                    dct = {
                        'found_terms': found_terms,
                        'total_terms': total_terms
                    }
                    super_category[category['category']] = dct
                else:
                    super_category.get(category['category'])['found_terms'].append(category['term'])
                    super_category.get(category['category'])['total_terms'] += nums

                full_dct['term_categories'][category['category']] = super_category[category['category']]
        if full_dct['term_categories'] != {}:
            stat.append(full_dct)

    for s in stat:
        print(s)


if __name__ == '__main__':
    text_paths = ['./test_data/texts/canidae.txt',
                  './test_data/texts/cedrus.txt',
                  './test_data/texts/forest_steppe.txt',
                  './test_data/texts/forest_tundra.txt',
                  './test_data/texts/jungle.txt',
                  './test_data/texts/meadow.txt',
                  './test_data/texts/pasture.txt',
                  './test_data/texts/pine.txt',
                  './test_data/texts/rosaceae.txt',
                  './test_data/texts/savanna.txt',
                  './test_data/texts/taiga.txt',
                  './test_data/texts/temperate_deciduous_forest.txt',
                  './test_data/texts/tundra.txt',
                  './test_data/texts/wolf.txt',
                  ]

    df, categories_info = build_ontology('./test_data/ontology.csv', [2,18])
    # print(categories_info)
    calculate_stat(text_paths, categories_info)

