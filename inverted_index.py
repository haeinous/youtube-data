#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""
import collections, re, nltk, pickle, sys
from nltk.stem.snowball import EnglishStemmer

from model import *
from server import app


###################################################################
# (1) Define classes for InvertedIndex, PostingsList, and Posting.
###################################################################

class PostingsList:
    """A singly linked list to store the postings."""

    def __init__(self, data=None):
        """data is a tuple with two elements: the unique doc_id and frequency."""
        if data:
            self.head = Posting(data)
            self.tail = self.head
        else:
            self.head = None
            self.tail = None

    def append(self, data):
        """Append a posting to the end of a linked list."""
        new_posting = Posting(data)

        if not self.head: # if the head is empty
            self.head = new_posting
        else:
            self.tail.next = new_posting      
        self.tail = new_posting
        new_posting.next = None

    def print_postings(self):
        """Print data for all postings."""
        current = self.head
        while current:
            print(current)
            current = current.next

    def __len__(self):
        i = 0
        current = self.head
        while current:
            i += 1
            current = current.next
        return i

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current
            current = current.next

    def __repr__(self):
        return '<PostingsList with {} postings>'.format(len(self))


class Posting:
    """A node in the PostingsList representing a token's frequency in a particular
    document. Its data is a two-element tuple: (doc_id, frequency)."""

    def __init__(self, data):
        """data is a tuple with two elements: the unique doc_id and frequency."""
        self.data = data
        self.next = None

    def __repr__(self):
        if self.next:
            return '<Posting: data={}>'.format(self.data)
        else:
            return '<Posting (tail): data={}, next={}>'.format(self.data, 
                                                               self.next)


class InvertedIndex(dict):
    """Inverted index data structure, consisting of a dictionary of all the unique 
    words pointing to a singly linked list of the documents in which it appears."""

    # Example: {('world'): [(1, 2), (3, 1), (5, 1)],
    #           ('hello'): [(3, 1), (4, 2)]}
    # 'world' appears 4 times across 3 docs (doc_id 1, 3, and 5) and twice in doc 1.

    def __init__(self):
        self.index = collections.defaultdict(list)

    def process_terms(self, document, document_id):
        """Process a term so it can be added to the index. The nltk module 
        provides a tokenizer function, word stemmer, as well as a list of 
        stopwords (English words to ignore)."""

        term = nltk.word_tokenize(document).lower()
        if term not in self.stopwords:
            term = self.stemmer.stem(term)
            frequency = term.count(document)
            if term not in self.index:
                self.index[term] = PostingsList((document_id, frequency))
            else:
                self.index[term].append((document_id, frequency))

    def print(self):
        """Print the inverted index."""
        print(dict(self.items()))

    def search(self, term):
        """Given a search term, return a list of documents containing it."""
        pass

    def __missing__(self, term):
        """Similar to defaultdict(list), if term doesn't exist, the term is added
        and it indicates that there are no matching documents."""

        self[term] = []
        return self[term]

    def __repr__(self):
        return '<InvIndex w/ {} terms:\n{}>'.format(len(self),
                                                    dict(self.items()))


################################################
# Generate the inverted index in the beginning.
################################################


def create_document_id(document_type, document_primary_key):
    """Create a new document in the documents table for a unique document_type/
    primary key pair."""

    document = Document(document_type=document_type,
                        document_primary_key=document_primary_key)
    db.session.add(document)

    try:
        db.session.commit()
    except:
        print('error creating document ID for {}'.format(document_primary_key))
        return

    document_id = Document.query.filter(Document.document_type == document_type
                               ).filter(Document.document_primary_key == document_primary_key
                               ).first(
                               ).document_id
    return document_id


def index_document(document_text, document_id, title=None):
    """Parse document text (and tokenize/stem if appropriate). Return a tuple where 
    index[0] is the token, index[1] is the doc ID, and index[2] is frequency."""

    if not title:
        good_tokens = generate_tokens(document_text)
        good_tokens.extend(produce_token_variations(good_tokens))

    else: # don't tokenize/stem for channel and video titles
        good_tokens = []
        good_tokens.append(document_text.lower())
        if title == 'channel': # add variations for entire channel title
            good_tokens.extend(produce_token_variations(good_tokens, channel=True))
        else:
            good_tokens.extend(produce_token_variations(good_tokens))            

    unique_tokens = set(good_tokens)
    stopwords = create_custom_stopwords()
    document_data = []

    for token in unique_tokens:
        if token not in stopwords:
            frequency = good_tokens.count(token)
            document_data.append((token, document_id, frequency))

    return document_data


def generate_inverted_index():
    """One-time operation after seeding data to generate an inverted index."""

    inverted_index = InvertedIndex() # instantiate inverted index

    # Inner function to add to the inverted index
    def add_data_to_inverted_index(document_data):
        """Assume document_data is a tuple where index[0] is the token string, index[1] 
        is the document ID, and index[2] is how frequently the token appears in the 
        document. Add tokens and postings to the inverted index."""

        for token_data in document_data:
            if token_data[0] not in inverted_index: # add a new token
                inverted_index[token_data[0]] = PostingsList((token_data[1], token_data[2]))
            else: # add Posting to the end of the PostingsList
                inverted_index[token_data[0]].append((token_data[1], token_data[2]))
    # - - end inner function - -

    # (1) Create document IDs
    i = 1
    for category in VideoCategory.query.all():
        if i:
            first_document_id = create_document_id('category', str(category.video_category_id))
            i = not first_document_id # changes i to False as long as first_document_id does not return None
        else:
            create_document_id('category', str(category.video_category_id))
    for channel in Channel.query.all():
        create_document_id('channel_title', channel.channel_id)
        create_document_id('channel_description', channel.channel_id)
    for video in Video.query.all():
        create_document_id('video_title', video.video_id)
        create_document_id('video_description', video.video_id)
    # for tags??
    for country in Country.query.all():
        create_document_id('country', country.country_code)


    # (2) Process document text (index_document function) and add to inverted index (inner function)

    documents_to_process = Document.query.filter(Document.document_id >= first_document_id
                                        ).filter(Document.document_primary_key.isnot(None)
                                        ).filter(Document.document_type.isnot(None)
                                        ).all() # doc IDs are autogenerated sequentially
    i = 0
    for document in documents_to_process:
        if i%100 == 0:
            print('{} out of {} documents added!'.format(i, len(documents_to_process)))

        if len(document.document_primary_key) == 11: # video
            video_object = Video.query.filter(Video.video_id == document.document_primary_key).first()
            if document.document_type == 'video_title':
                document_data = index_document(video_object.video_title,
                                               document.document_id,
                                               title='video')
                add_data_to_inverted_index(document_data)
            else:
                document_data = index_document(video_object.video_description, 
                                               document.document_id)
                add_data_to_inverted_index(document_data)

        elif len(document.document_primary_key) == 24: # channel
            channel_object = Channel.query.filter(Channel.channel_id == document.document_primary_key).first()
            if document.document_type == 'channel_title':
                document_data = index_document(channel_object.channel_title,
                                               document.document_id,
                                               title='channel')
                add_data_to_inverted_index(document_data)
            else:
                document_data = index_document(channel_object.channel_description, 
                                               document.document_id)
                add_data_to_inverted_index(document_data)

        elif document.document_type == 'category':
            category_name = VideoCategory.query.filter(VideoCategory.video_category_id == 
                                                       int(document.document_primary_key) # bec document_primary_keys were stored in the db as strings
                                              ).first(
                                              ).category_name
            document_data = index_document(category_name, # supplement with synonyms?
                                           document.document_id)
            add_data_to_inverted_index(document_data)

        elif document.document_type == 'country':
            country_name = Country.query.filter(Country.country_code == 
                                                document.document_primary_key
                                       ).first(
                                       ).country_name # supplement with alternate names?
            document_data = index_document(country_name, 
                                           document.document_id)
            add_data_to_inverted_index(document_data)

        # elif tags tk tk tk

    print('Done generating inverted index.')
    return inverted_index


######################################
# Functions to improve token quality.
######################################

def create_custom_stopwords():
    """Use the NLTK module's english stopwords and combine it with custom YouTube
    stopwords. Return the stopwords set."""

    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    with open('additional_stopwords.txt') as f:
        additional_stopwords = f.read().split()
    for word in additional_stopwords:
        stopwords.add(word)

    return stopwords


def generate_tokens(document_text):
    """Process text so it can be added to the index. The NLTK module provides a 
    tokenizer, word stemmer, and a list of stopwords."""

    stemmer = EnglishStemmer() # from NLTK module

    tokens = [additional_pruning(process_url_prefixes(stemmer.stem(token.lower()))) 
              for token in nltk.word_tokenize(document_text) if len(token) > 2]

    good_tokens = []
    for token in tokens:
        good_tokens.extend([custom_youtube_stemmer(item) 
                            for item in process_url_prefixes(token)])

    better_tokens = []
    for token in good_tokens:
        if loop_through_chars(token):
            better_tokens.extend(process_url_ends(token))  # can you use list comprehension with extend?

    return better_tokens


def process_url_ends(token):
    if len(token) > 3 and (token[-4:] == '.com' or 
                           token[-4:] == '.org' or
                           token[-4:] == '.net' or
                           token[-6:] == '.co.uk'):

        tokens = [item for item in token.split('.') 
                  if item not in {'', 'com', 'org', 'net', 'co', 'uk'}]
    else:
        tokens = [token]

    return tokens


def process_url_prefixes(token):
    if token[:2] == '//':
        token = token[2:]
    if token[:4] == 'www.':
        token = token[4:]
    
    tokens = [additional_pruning(item) for item 
              in token.split('/')[0] if item and '=' not in item]

    return tokens


def custom_youtube_stemmer(token):
    if token == 'twitter.com':
        token = 'twitter'
    elif token == 'facebook.com':
        token = 'facebook'
    elif token == 'instagram.com':
        token = 'instagram'
    elif token == 'patreon.com':
        token = 'patreon'
    elif token == 'reddit.com':
        token = 'reddit'
    elif token == 'soundcloud.com':
        token = 'soundcloud'
    elif token == 'musical.li' or token == 'musically':
        token = 'musical.ly'
    elif token == 'pinterest.co.uk':
        token = 'pinterest'
    elif token == 'yahoo.co.uk' or token == 'yahoo.com':
        token = 'yahoo'
    elif token == 'e-mail':
        token = 'email'
    elif token == 'teespr' or token == 'teespring.com' or token == 'tspr.ng':
        token = 'teespring'
    elif token == 'vlogs' or token == 'vlogger' or token == 'vloggers':
        token = 'vlog'
    elif token == 'right-w':
        token = 'right-wing'
    elif token == 'left-w':
        token = 'left-wing'
    elif token == 't-shirt':
        token = 'tshirt'
    elif token == 'merchandis':
        token = 'merch'
    elif token == 'blogger':
        token = 'blog'
    elif token == 'live-stream':
        token = 'livestream'

    return token


def additional_pruning(token):
    """Use regex to adjust tokens beginning/ending in non-alphanumeric characters."""

    if not token[-1].isalnum():
        token = re.search(r'[\w-]+(?=[\W]+)', token).group(0)
    if not token[0].isalnum():
        token = re.search(r'\w+', token).group(0)

    return token


def loop_through_chars(token):
    """Loop through all the chars in token and remove those that: (a) don't have 
    a single alpha character or (b) it begins/ends with a non-latin character. 
    Return bool to determine whether to further process."""

    at_least_one_alpha = False

    for i in range(len(token)):
        if i == len(token) or i == 0:
            if token[i] > 'á': # if non-latin alpha character
                return False
        while not at_least_one_alpha:
            if token[i].isalpha():
                at_least_one_alpha = True

    if at_least_one_alpha:
        return True
    else:
        return False


def produce_token_variations(tokens, channel=False):
    """Return a list split and joined token variations. For example, for the token 
    'kiera bridget', 'kierabridget', 'kiera', and 'bridget' would be added."""

    token_variations = []

    for token in tokens:
        variations = [variation for variation in re.split(r'\W+', token) if variation] # split token on non-alphanumeric chars
        if len(variations) > 1: # there was actually something to split on
            token_variations.extend(variations) # append, for example, 'kiera' and 'bridget'
            if channel:
                token_variations.append(''.join(variations)) #append 'kierabridget' but only for channels

    return token_variations


###################################################################
# Helper functions for dealing with the inverted index.
###################################################################


def append_to_inverted_index(new_document):
    """Add data from new documents submitted via the /add-data route to the
    inverted index.""" 

    pass # can you unpickle an index in append mode?


def sort_tokens_by_freq():
    """Return a list of tokens, reverse-sorted (for popping in constant time)
    by how frequently they appear in documents."""

    inverted_index = load_inverted_index()
    tokens_by_freq = []

    for token, postings_list in inverted_index.items():
        tokens_by_freq.append((token, len(postings_list)))

    return sorted(tokens_by_freq, key=lambda x: x[1])


def pickle_inverted_index(inverted_index):
    """Dump the inverted index into a pickle so it doesn't need
    to be regenerated every time."""

    sys.setrecursionlimit(3000) # exceed the default max recursion limit
    with open('inverted_index.pickle', 'wb') as f:
        pickle.dump(inverted_index, f)


def load_inverted_index():
    """Load the pickled inverted index."""

    with open('inverted_index.pickle', 'rb') as f:
        return pickle.load(f)



if __name__ == '__main__':

    connect_to_db(app)
    app.app_context().push()

    # pickle_inverted_index(generate_inverted_index())
