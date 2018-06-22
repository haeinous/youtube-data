#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""

import datetime, random, collections, re, pickle, sys, math
import numpy as np
from jinja2 import StrictUndefined
from flask import Flask, render_template, request, flash, redirect, session, jsonify, Markup
from flask_debugtoolbar import DebugToolbarExtension
from sqlalchemy import func

from model import *
from api_youtube import *
from all_api_calls import *
from inverted_index import *

app = Flask(__name__)

# Required to use Flask sessions and the debug toolbar
app.secret_key = 'ABC'

# Raises an error when using an undefined variable in Jinja.
app.jinja_env.undefined = StrictUndefined

##### classes

class Trie:
    """A trie abstract data structure."""
    def __init__(self):
        self.root = TrieNode('')

    def add_word(self, word, freq):
        """Adds word to a trie if it doesn't exist, creating new nodes or 
        updating frequencies where necessary."""

        current = self.root

        for char in word:
            if char in current.children:
                current = current.children[char]
            else:
                current.add_child(char)
                current = current.children[char]

        current.freq = freq

    def find_prefix_node(self, prefix):
        """Given a prefix string, return the trie node of the prefix's last 
        character, False if it does not exist."""

        current = self.root

        for char in prefix:
            if char in current.children:
                current = current.children[char]
            else:
                return False

        return current

    def __repr__(self):
        return '<Trie root={}>'.format(self.root)


class TrieNode:
    """A trie node containing one character of a word. If self.freq == 0, it
    is not the end of a word. When self.freq > 0, it signifies the end of the word
    and also conveys frequency (i.e., relevancy)."""

    def __init__(self, value):
        self.value = value
        self.children = dict() # key is char, value is TrieNode object
        self.freq = 0

    def add_child(self, value):
        child_node = TrieNode(value)
        self.children[value] = child_node

    def list_words(self, previous=None):
        """Return a list of all valid words under and including the trie node;
        return [] if it does not exist."""

        all_words = []

        if not previous: # if it's the first recursive loop
            previous = ''

        if self.freq:
            all_words.append((previous, self.freq))

        for char, node in self.children.items():
            for word_and_freq in list_words(node, previous=previous+char):
                all_words.append(word_and_freq)

        return all_words

    def __repr__(self):
        return '<TrieNode {} ({})\nch: {}>'.format(self.value,
                                                   self.freq,
                                                   self.children)


class RandomBag:
    """A random bag that allows elements to be inserted and selected randomly
    in O(1) time."""

    def __init__(self):
        self.dictionary = dict() # for storing key(object)-value(index in list)
        self._list = list() # list of objects (or should I use a doubly linked list like deque?)
        self.length = 0

    def insert(self, object):
        self.dictionary[object] = self.length - 1 # index of object in list
        self._list.append(object)
        self.length += 1

    def get_random_elements(self, number):
        """Return a number of random elements without permanently removing them."""
        random_elements = []

        i = 1
        for num in range(number):
            r = random.randint(0, self.length - i)
            self._list[-1], self._list[r] = self._list[r], self._list[-1] # O(1) to remove last element
            random_elements.append(self._list.pop())
            i += 1

        for element in random_elements:
            self._list.append(element)
            self.dictionary[element] = self.length - 1

        return random_elements

    def __len__(self):
        return len(self.dictionary) # or it could may well be len(self._list) as they're both O(1)

    def __repr__(self):
        return '<RandomBag containing {} item(s).>'.format(self.length)


class Vertex:
    """A vertex in a graph."""
    def __init__(self, value, neighbors=None):
        self.value = value
        self.name = value.video_id

        if not neighbors:
            self.neighbors = {}
        else:
            self.neighbors = neighbors

    def get_all_neighbors(self, include_weight=False):
        if self.include_weight:
            return set(self.neighbors.items()) # set of tuples
        else:
            return set(self.neighbors.keys())

    def get_closest_neighbors(self, num_neighbors):
        """Return a list num_neighbors long of the vertex's closest neighbors by weight."""
        assert num_neighbors > len(self.neighbors), 'num_neighbors too large (max is {}).'.format(len(self.neighbors))
        
        return sorted(list(self.get_all_neighbors(include_weight=True)), 
                           key=lambda x: (-x[1], x[0]))[:num_neighbors]

    def add_neighbor(self, vertex, weight):
        assert isinstance(vertex, Vertex), 'The neighbor you want to add is not a vertex.'
        self.neighbors[vertex] = weight

    def get_edge_weight(self, neighbor_vertex):
        assert isinstance(neighbor_vertex, Vertex), 'The neighbor must be an instance of the Vertex class.'
        return self.neighbors[neighbor_vertex]

    def __repr__(self):
        return '<Vertex {} has {} neighbors>'.format(self.name,
                                                     len(self.neighbors))


class Graph:
    """An undirected graph consisting of vertices and edges.
    Implemented using an adjacency list given its sparse nature."""

    def __init__(self, vertices=None):
        """Assume vertices is a dictionary whose key is the vertex name and whose
        value is the Vertex object."""

        if not vertices:
            self.vertices = collections.defaultdict(None, dict())
            self.num_vertices = 0
        else:
            self.vertices = vertices
            self.num_vertices = len(vertices)

    def get_vertices(self):
        """Return a set containing the names of all vertices within the graph."""
        return set(self.vertices.keys())

    def get_all_vertices(self):
        return self.vertices.items()

    def add_vertex(self, vertex):
        assert isinstance(vertex, Vertex), 'The argument needs to be a Vertex object.'
        self.vertices[vertex.name] = vertex
        self.num_vertices += 1

    def add_edge(self, vertex1_name, vertex2_name, weight):
        if vertex1_name not in self.vertices:
            self.add_vertex(vertex1_name)
        if vertex2_name not in self.vertices:
            self.add_vertex(vertex2_name)
        vertex1_name.add_neighbor(vertex2_name, weight)

    def __repr__(self):
        return '<Graph with {} vertices: {}>'.format(len(self.vertices),
                                                     self.vertices)

def calculate_percent_demonetized(channel_id):
    """Given a channel_id, return the % of videos in that channel that are demonetized."""

    all_videos = make_integer(db.session.query(func.count(Video.video_id)
                                                   ).filter(Video.channel_id == channel_id
                                                   ).first())
    demonetized_videos = make_integer(db.session.query(func.count(Video.video_id)
                                                           ).filter(Video.channel_id == channel_id
                                                           ).filter(Video.is_monetized == False
                                                           ).first())
    return round(demonetized_videos/all_videos * 100)


def calculate_weight_by_shared_tags(youtube_id1, youtube_id2):
    """Given two YouTube IDs of the same type (video ID or channel ID), return
    the number of shared tags between videos or channels."""

    assert len(youtube_id1) == len(youtube_id2), 'Arguments need to be the same type (either video_id or channel_id).'
    
    if len(youtube_id1) == 11: # video ID
        video_tags1 = {tag.tag for tag in Tag.query.join(TagVideo
                                                  ).filter(TagVideo.video_id.match(youtube_id1)
                                                  ).all()}
        video_tags2 = {tag.tag for tag in Tag.query.join(TagVideo
                                                  ).filter(TagVideo.video_id.match(youtube_id2)
                                                  ).all()}
        return abs(len(video_tags1 & video_tags2))

    else:
        channel_tags1 = {tag.tag for tag in Tag.query.join(TagVideo
                                                    ).join(Video
                                                    ).filter(TagVideo.video_id.in_(
                                                        [video.video_id for video in Video.query.filter(Video.channel_id.match(youtube_id1) 
                                                        ).all()])
                                                    ).all()}
        channel_tags2 = {tag.tag for tag in Tag.query.join(TagVideo
                                                    ).join(Video
                                                    ).filter(TagVideo.video_id.in_(
                                                        [video.video_id for video in Video.query.filter(Video.channel_id.match(youtube_id2) 
                                                        ).all()])
                                                    ).all()}
        return abs(len(channel_tags1 & channel_tags2))


def generate_video_graph(channel_id):
    """Given a channel_id, generate a graph of all of its videos where weight
    is the number of shared tags between them."""

    # 1. Instantiate vertices and graph
    all_videos = Video.query.filter(Video.channel_id == channel_id).all()
    all_vertices = {}

    for video in all_videos:
        all_vertices[video.video_id] = Vertex(video)

    video_graph = Graph(all_vertices) # at this point none of the vertices are connected.
    
    # 2. Add edges/connections between vertices
    video_tags = {} # key is video_id, value is set of all tags
    for video in all_videos:
        all_tags = set(tag.tag for tag in Tag.query.join(TagVideo
                                                  ).filter(TagVideo.video_id == video.video_id
                                                  ).all())
        video_tags[video.video_id] = all_tags

    # 3. filter out tags that are in every single video (e.g., channel name)
    tags_in_all_videos = set()

    tags_so_far = set()
    i = 0
    for video in video_tags:
        if i == 0:
            tags_so_far = video_tags[video]
        else:
            tags_so_far = tags_so_far & video_tags[video]
        i += 1

    for video in video_tags:
        video_tags[video] -= tags_so_far

    video_ids = set([video.video_id for video in all_videos])
    other_video_ids = set([other_video.video_id for other_video in all_videos])

    processed_video_pairs = set()
    for video_id in video_ids:
        for other_video_id in other_video_ids:
            if video_id != other_video_id and (video_id, other_video_id) not in processed_video_pairs:
                edge_weight = len(video_tags[video_id] & video_tags[other_video_id])
                if edge_weight:
                    video_graph.add_edge(all_vertices[video_id], 
                                         all_vertices[other_video_id], 
                                         edge_weight)        
        processed_video_pairs.add((video_id, other_video_id))
        processed_video_pairs.add((other_video_id, video_id))

    return video_graph


@app.route('/video-graph-by-shared-tags.json')
def generate_video_graph_by_shared_tags():
    """Return JSON to allow D3 to illustrate shared tags and monetization status."""

    data = {'nodes': [], 'links': []}

    channel_id = request.args.get('channelId')

    video_graph = generate_video_graph(channel_id)
    all_vertices = {}
    for item in video_graph.get_all_vertices():
        all_vertices[item[0]] = item[1]
    processed_pairs = set()

    for vertex in all_vertices:
        if all_vertices[vertex].value.is_monetized:
            data['nodes'].append({'id': vertex, 'group': 1})
        else:
            data['nodes'].append({'id': vertex, 'group': 2})
        for neighbor in all_vertices[vertex].neighbors:
            if (vertex, neighbor) not in processed_pairs:
                data['links'].append({'source': vertex,
                                      'target': neighbor.name,
                                      'value': all_vertices[vertex].neighbors[neighbor]})
                processed_pairs.add((vertex, neighbor.name))
                processed_pairs.add((neighbor.name, vertex))

    return jsonify(data)

def get_document_text(document_id):

    doc_type, doc_object = get_original_document_object(document_id)

    if doc_type == 'video':
        return doc_object.video_title
    elif doc_type == 'channel':
        return doc_object.channel_title
    elif doc_type == 'category':
        return doc_object.category_name
    else:
        return None

def get_original_document_object(document_id):
    """Given a document_id, return an object for the underlying document (i.e.,
    a Video object if the document were a video description, for example."""

    document_object = db.session.query(Document).get(document_id)

    if len(document_object.document_primary_key) == 11:
        result_type = 'video'
        video = Video.query.get(document_object.document_primary_key)
        return [result_type, video]
    elif len(document_object.document_primary_key) == 24:
        result_type = 'channel'
        channel = Channel.query.get(document_object.document_primary_key)
        return [result_type, channel]
    elif document_object.document_type == 'category':
        result_type = 'category'
        category = VideoCategory.query.get(document_object.document_primary_key)
        return [result_type, category]


############################################################################################
# See inverted_index.py for details on the InvertedIndex, PostingsList, and Posting classes.
############################################################################################

"""Sincere thanks to Rebecca Weiss and Christian S. Perone for creating easy-to-understand
and sufficiently advanced online material to teach me enough linear algebra to fully understand the 
vector operations behind tf-idf. In particular, I found these resources very helpful:

> blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii
> stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/tfidf.html

Also, I stumbled upon a computational linguistics class taught by one Chris Callison-Burch at UPenn. 
Working through the vector space model assignment (week 3) reminded me that I deeply enjoy (and am 
good at!) math. I'm excited to continue to work through the syllabus.

Another shoutout to Christopher Manning / Prabhakar Raghavan / Hinrich Schütze / Stanford (and maybe
Cambridge University Press?) for allowing the highly readable "Intro to Information Retrieval" textbook
to exist online for free."""

def l2_normalizer(vector):
    """Normalize vectors to L2 Norm = 1 (euclidian normalized vectors) to compensate
    for the effect of document length (magnitude)."""

    denominator = np.sum([item**2 for item in vector])

    try:
        normalized_vector = [round(item/math.sqrt(denominator), 3) for item in vector]
    except ZeroDivisionError: # all items are 0
        normalized_vector = vector

    return normalized_vector


def generate_relevant_doc_ids(query_tokens):
    """Given a set of unique tokens, create a list of document IDs that
    represent documents that should be ranked for relevance to the query vector.""" 

    with open('inverted_index_real.pickle', 'rb') as f:
        inverted_index = pickle.load(f)

    all_document_ids = []

    for token in query_tokens:
        try:
            postings_list = inverted_index[token]
        except:
            pass
        else:
            all_document_ids.extend([posting.data[0] for posting in postings_list])

    return list(set(all_document_ids))


def construct_idf_matrix(idf_vector):
    """Build an idf matrix out of an idf vector in order to normalize all the
    tf vectors by normalizing / mulitiplying them with the idf matrix."""

    idf_matrix = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_matrix, idf_vector)

    return idf_matrix


# A lot goes on under this function to obviate the need to constantly unpickle
# the inverted index.
def construct_tf_idf_matrix(query_tokens, all_document_ids):
    """Construct a tf-idf matrix with which to normalize idf across the query vector 
    and relevant documents identified from the inverted index."""

    with open('inverted_index_real.pickle', 'rb') as f:
        inverted_index = pickle.load(f)

    # - - - - - inner function no.1 - - - - - -
    def construct_query_vector(query_tokens, ordered_query_tokens):
        """Treat the query as its own document and normalize it with the other relevant
        documents in order to calculate cosine similarity to rank documents in terms
        of relevance."""

        query_vector = []

        for token in ordered_query_tokens:
            if inverted_index[token]: # if it exists
                query_vector.append(ordered_query_tokens.count(token))
            else:
                query_vector.append(0)

        return query_vector
    # - - - - - - inner function no.2 - - - - - -
    def construct_tf_vector(document_id):
        """Build a term freq vector for a certain document."""

        document_text = get_document_text(document_id)

        tf_vector = []

        document_tokens = generate_tokens(document_text)

        for token in ordered_query_tokens:
            tf_vector.append(document_tokens.count(token))
        return tf_vector  
    # - - - - - - inner function no.3 - - - - - -
    def calculate_idf(token):
        """Calculate a token's inverse document frequency."""

        return np.log(len(inverted_index)/(1+len(inverted_index[token]))) # +1 protects against zero division

    # - - - - - - end inner functions - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - -
# 1 Establish a list unique and ordered query tokens
# - - - - - - - - - - - - - - - - - - - - - - - - -
    ordered_query_tokens = list(query_tokens)

# - - - - - - - - - - - - - - - - - -
# 2 Construct L2-normalized tf vectors
# - - - - - - - - - - - - - - - - - -
    # 2(a) Construct tf query vector.
    query_vector = l2_normalizer(construct_query_vector(query_tokens, ordered_query_tokens))

    # 2(b) Construct tf vectors for all other documents
    all_document_vectors =[]

    for document_id in all_document_ids:
        all_document_vectors.append(l2_normalizer(construct_tf_vector(document_id)))

    # all_document_vectors, a list, is a tf matrix with row-wise L2 norms of 1
    # (i.e., term frequencies are normalized within docs).

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3 Construct an idf vector to normalize tf across docs through idf weighting.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    idf_vector = [calculate_idf(token) for token in ordered_query_tokens]
#   > 3(b) Build an idf matrix to maintain vector operations (vs. scalar, which is what 
#          would happen if we just multiplied two vectors of the same magnitude and direction).
    idf_matrix = construct_idf_matrix(idf_vector)

#   > 3(c) Normalize term frequencies across documents by multiplying tf vectors by the
#          idf matrix via dot product. Reapply euclidian normalization to each 
#          tf vector so L2 norm remains 1.
    tf_idf_matrix = []

    for tf_vector in all_document_vectors:
        tf_idf_matrix.append(l2_normalizer(np.dot(tf_vector, idf_matrix)))
    
#   > 3(d) Normalize the query vector too!
    query_vector = l2_normalizer(np.dot(query_vector, idf_matrix))

    return query_vector, tf_idf_matrix


def calculate_cosine_similarity(query_vector, other_document_vector):
    """Calculate cosine similarity between the query vector and document vectors by 
    taking the dot product. A lower value (smaller angle) indicates higher similarity and
    greater relevance.
    cosine  = ( v1 * v2 ) / ||v1|| x ||v2||  """

    return round(np.dot(query_vector, other_document_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(other_document_vector)), 3)

    """
    n.b.: A lot of this can be done much more easily by going through thescikit-learn 
    module, which has tools like tf-idf vectorizers and cosine similarity functions.
    I opted to do it the hard way once (to learn, and to build character).

    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> from sklearn.metrics.pairwise import cosine_similarity
    """

def organize_search_results(documents_by_relevance):
    """Given a list of documents ordered by search result ranking, query the
    database for additional information to display on the client side."""

    document_data = {'video': [], 'channel': []}

    for document in documents_by_relevance:
        result_type, db_object = get_original_document_object(document[1]) # document[1] = doc_id        
        if result_type != 'category':
            document_data[result_type].append([db_object, document[0]]) # document[0] = cosine_similarity

    return document_data


@app.route('/search', methods=['GET'])
def display_search_results():
    """Display search results; do linear algebra."""

# - - - - - - - - - - - - - -
# 1 Process the search query:
# - - - - - - - - - - - - - - 
#   > 1(a) Get query from client
    search_query = request.args.get('q')
    print('search_query: ' + search_query)

#   > 1(b) Tokenize it
    query_tokens = set(generate_tokens(search_query))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2 Look through the inverted_index for relevant documents, if any.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#   > 2(a) Unpickle the inverted index
    with open('inverted_index_real.pickle', 'rb') as f:
        inverted_index = pickle.load(f)

#   > 2(b) Gather relevant document IDs (if any)
    all_document_ids = generate_relevant_doc_ids(query_tokens)

#   > 2(c) Make sure it's not an empty list (if so return None)
    if not all_document_ids:
        return render_template('search.html', search_results=None,
                                              search_query=search_query)
    else:
# - - - - - - - - - - - - - - - - - - - - - - -
# 3 Create tf vectors and matrices.
# - - - - - - - - - - - - - - - - - - - - - - -
#    > 3(a) Build normalized tf vectors for all docs (including the query) and
#           a tf-idf matrix to determine the most relevant documents based on 
#           cosine similarity.
        query_vector, tf_idf_matrix = construct_tf_idf_matrix(query_tokens, all_document_ids)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 4 Compare the normalized query vector with all other normalized tf-idf vectors.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    document_vectors_and_ids = list(zip(tf_idf_matrix, all_document_ids))

    docs_by_cosine_similarity = []
    for document_item in document_vectors_and_ids:
        cosine_similarity = round(calculate_cosine_similarity(query_vector, document_item[0]), 3)
        docs_by_cosine_similarity.append([cosine_similarity, document_item[1]]) # tuple: (cosine_similarity, doc_id)

    docs_by_cosine_similarity.sort()
    print('docs_by_cosine_similarity={}'.format(docs_by_cosine_similarity))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 5 Gather additional info from the database to feed to the search results page.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    try:
        document_data = organize_search_results(docs_by_cosine_similarity[:20])
    except KeyError: # if fewer than 20 results
        document_data = organize_search_results(docs_by_cosine_similarity)

    search_results = document_data

    print(search_results)
    return render_template('search.html', search_results=search_results,
                                          search_query=search_query)


##### Helper functions

def make_integer(sqlalchemy_object):
    """Assume object is a sqlalchemy.util._collections.result.
    Return an integer."""

    return int(str(sqlalchemy_object)[1:-2])


def format_timedelta(timedelta_object):
    """Assume timedelta_object represents video duration.
    Return a string representing the video duration in minutes, 
    seconds, and hours (if applicable)."""

    hours, minutes, seconds = map(int, str(timedelta_object).split(':'))

    if hours == 0:
        if minutes == 0:
            return '{} seconds'.format(seconds)
        else:
            return '{}:{}'.format(minutes, seconds)
    else:
        return '{}:{}:{}'.format(hours, minutes, seconds)


##### /end helper functions


##### Normal flask routes

@app.route('/')
def index():
    """Homepage."""

    videos_in_db = make_integer(db.session.query(func.count(Video.video_id)
                                         ).first())
    channels_in_db = make_integer(db.session.query(func.count(Channel.channel_id)
                                           ).first())

    return render_template('homepage.html', videos_in_db=videos_in_db,
                                            channels_in_db=channels_in_db)


def calculate_demonetization_percentage_by_tag(tag):
    """Given a tag, return the percentage of videos with that tag that have
    been demonetized."""

    tag = tag.lower().strip()

    all_videos = TagVideo.query.join(Tag
                              ).filter(Tag.tag == tag
                              ).all()
    demonetized_videos = TagVideo.query.join(Tag
                                ).filter(Tag.tag == tag
                                ).join(Video
                                ).filter(Video.is_monetized == False
                                ).all()

    return round(len(demonetized_videos)/len(all_videos)*100)


@app.route('/get-individual-tag-data.json')
def generate_individual_tag_data_json():
    """Query the database the retrieve tag demonetization data and
    return a json string."""

    tag = request.args.get('newTag')

    json_response = {'labels': [tag],
                     'datasets': [{'backgroundColor': [], # populate on client side
                                   'data': []}]}

    demonetization_percentage = calculate_demonetization_percentage_by_tag(tag)
    json_response['datasets'][0]['data'].append(demonetization_percentage)

    print(json_response)
    return jsonify(json_response)


@app.route('/get-tag-data.json')
def generate_tag_data_json():
    """Query the database the retrieve tag demonetization data and
    return a json string."""

    tags = ['donald trump', 'hillary clinton', 'bernie sanders']

    json_response = {'labels': [],
                     'datasets': [{'backgroundColor': [], # populate on client side
                                   'data': []}]}
    for tag in tags:
        json_response['labels'].append(tag)
        demonetization_percentage = calculate_demonetization_percentage_by_tag(tag)
        json_response['datasets'][0]['data'].append(demonetization_percentage) 
    
    return jsonify(json_response)


@app.route('/get-individual-tag-data-for-chart.json')
def generate_tag_data_for_individual_tag_chart():
    """Given an individual tag, load demonetization data and # of videos using it."""

    tag = request.args.get('tag')
    tag_id = TagVideo.query.filter(Tag.tag == tag).first().tag_id

    total_videos = db.session.query(TagVideo
                            ).join(Tag
                            ).filter(Tag.tag == tag
                            ).count()

    demonetized_videos = db.session.query(TagVideo
                                  ).join(Tag
                                  ).filter(Tag.tag == tag
                                  ).join(Video
                                  ).filter(Video.is_monetized == False
                                  ).count()

    len(TagVideo.query.filter(TagVideo.tag_id == tag_id
                                          ).join(Video
                                          ).filter(Video.is_monetized == False
                                          ).all())

    percent_demonetized = round(demonetized_videos/total_videos * 100)

    json_response = {'labels': [tag],
                     'datasets': [{'label': '% monetized',
                                   'backgroundColor': 'rgba(50, 178, 89, 1)',
                                   'data': [100-percent_demonetized]},
                                  {'label': '% demonetized',
                                   'backgroundColor': 'rgba(238, 39, 97, 1)',
                                   'data': [percent_demonetized]}
                                 ],
                     'total_videos': total_videos}

    return jsonify(json_response)


@app.route('/about')
def about_page():
    """About page."""

    videos_in_db = make_integer(db.session.query(func.count(Video.video_id)
                                         ).first())
    channels_in_db = make_integer(db.session.query(func.count(Channel.channel_id)
                                           ).first())

    return render_template('about.html', videos_in_db=videos_in_db,
                                         channels_in_db=channels_in_db)


@app.route('/explore/', methods=['GET'])
def explore_page():

    return render_template('explore.html')


@app.route('/explore/channels/', methods=['GET'])
def show_channels_page():
    """Show a random selection of YouTube channels."""

    videos_in_db = make_integer(db.session.query(func.count(Video.video_id)
                                         ).first())

    channels = RandomBag()
    for channel in Channel.query.all(): # create RandomBag of channels
        channels.insert(channel)
    
    random_channels = channels.get_random_elements(16)
    channel_videos = []

    for channel in random_channels:
        video = Video.query.filter(Video.channel_id == channel.channel_id
                          ).filter(Video.thumbnail_url.isnot(None)
                          ).first()
        channel_videos.append(video)

    random_channels = list(zip(random_channels, channel_videos))

    return render_template('channels.html',
                            videos_in_db=videos_in_db,
                            random_channels=random_channels)


@app.route('/explore/channels/<channel_id>')
def show_specific_channel_page(channel_id):
    """Show info about a specific YouTube channel."""

    # get channel data
    channel = Channel.query.get(channel_id)
    channel_country = Country.query.filter(Country.country_code 
                                           == channel.country_code
                                  ).first().country_name
    try:
        add_channel_stats_data(parse_channel_data(get_info_by_youtube_id(channel_id), channel_in_db=True))    
    except:
        pass

    channel_stats = ChannelStat.query.filter(ChannelStat.channel_id == channel_id
                                    ).order_by(ChannelStat.retrieved_at.desc()
                                    ).first()
    # get video data
    videos = Video.query.filter(Video.channel_id == channel_id
                       ).order_by(Video.published_at.desc()).all()
    demonetized_videos = Video.query.filter(Video.channel_id == channel_id
                                   ).filter(Video.is_monetized == False
                                   ).filter(Video.video_status.is_(None)
                                   ).order_by(Video.published_at.desc()
                                   ).all()

    if len(demonetized_videos) == 0:
        demonetization_percentage = 0
    else:
        if len(demonetized_videos) > 3:
            demonetized_videos = demonetized_videos[:4]
        else:
            demonetized_videos = demonetized_videos

        try:
            demonetization_percentage = len(demonetized_videos)/len(videos)
        except ZeroDivisionError:
            demonetization_percentage = 0
        else:
            if demonetization_percentage > 0 and demonetization_percentage < .01:
                demonetization_percentage = 1
            else:
                demonetization_percentage = round(demonetization_percentage * 100)

    return render_template('channel.html',
                            channel=channel,
                            channel_stats=channel_stats,
                            channel_country=channel_country,
                            videos=videos,
                            demonetized_videos=demonetized_videos,
                            videos_in_db=len(videos),
                            demonetization_percentage=demonetization_percentage)


@app.route('/explore/videos/')
def show_videos_page():
    """Show a random selection of YouTube videos."""

    videos_in_db = make_integer(db.session.query(func.count(Video.video_id)
                                         ).first())

    monetized_videos = RandomBag()
    demonetized_videos = RandomBag()

    for video in Video.query.filter(Video.is_monetized == True
                           ).filter(Video.video_status == None
                           ).all():
        monetized_videos.insert(video)
    for video in Video.query.filter(Video.is_monetized == False
                           ).filter(Video.video_status == None
                           ).all():
        demonetized_videos.insert(video)
    
    random_monetized_videos = monetized_videos.get_random_elements(8) 
    random_demonetized_videos = demonetized_videos.get_random_elements(8)
    random_videos = (random_monetized_videos[:4] + 
                     random_demonetized_videos[:4] +
                     random_monetized_videos[4:] + 
                     random_demonetized_videos[4:])

    return render_template('videos.html',
                            videos_in_db=videos_in_db,
                            random_videos=random_videos)


@app.route('/explore/videos/<video_id>')
def show_specific_video_page(video_id):
    """Show info about a specific video."""

    video = Video.query.filter(Video.video_id == video_id).first()
    thumbnail_url = video.thumbnail_url

    # Update the video_stats table with the most up-to-date info
    try:
        add_video_stats_data(parse_video_data(get_info_by_youtube_id(video_id)))    
    except:
        pass

    # Get video_stats
    video_stats = VideoStat.query.filter(VideoStat.video_id == video_id
                                ).order_by(VideoStat.retrieved_at.desc()
                                ).first()

    channel = Channel.query.join(Video).filter(Video.video_id == video_id).first()
    image_analysis = ImageAnalysis.query.filter(ImageAnalysis.video_id == video_id).first()
    if image_analysis:
        nsfw_score = round(image_analysis.nsfw_score)
    else:
        nsfw_score = None
    text_analyses = TextAnalysis.query.filter(TextAnalysis.video_id == video_id).all()
    text_analyses = [(text_analysis.textfield, 
                      text_analysis.sentiment_score, 
                      text_analysis.sentiment_magnitude) for text_analysis in text_analyses] # list comprehension
    if Tag.query.join(TagVideo).filter(TagVideo.video_id == video_id).first():
        tags = Tag.query.join(TagVideo).filter(TagVideo.video_id == video_id).all()
    else:
        tags = []

    category = VideoCategory.query.join(Video).filter(Video.video_id == video_id).first()
    duration = format_timedelta(video.duration)

    return render_template('video.html',
                            video=video,
                            thumbnail_url=thumbnail_url,
                            video_stats=video_stats,
                            channel=channel,
                            image_analysis=image_analysis,
                            nsfw_score=nsfw_score,
                            text_analyses=text_analyses,
                            tags=tags,
                            category=category,
                            duration=duration)


@app.route('/explore/categories/')
def show_categories_page():

    categories = VideoCategory.query.all()

    return render_template('categories.html',
                            categories=categories)


@app.route('/category-data.json')
def generate_category_chart_data():

    colors = ['rgba(238, 39, 97, 1)', # pink #ee2761
              'rgba(40, 37, 98, 1)', # purple #282562
              'rgba(50, 178, 89, 1)', # green #32b259
              'rgba(94, 200, 213, 1)', # blueberry #5ec8d5
              'rgba(255, 242, 0, 1)', #yellow #fff200
              'rgba(188, 0, 141, 1)', #magenta #BC008D
              'rgba(43, 126, 195, 1)', # blue #2B7EC3
              'rgba(255, 106, 42, 1)', # orange #FF6A2A
              'rgba(33, 201, 133, 1)', # emerald #21C985
              'rgba(255, 201, 0, 1)', # light yellow-orange #FFC900
              'rgba(171, 230, 224, 1)', #pale blue #ABE6E0
              'rgba(255, 204, 204, 1)', # pale pink #FFCCCC
              'rgba(255, 242, 204, 1)', # pale yellow #FFF2CC
              'rgba(203, 254, 203, 1)', # pale green #CBFECB
              'rgba(218, 215, 217, 1)'] # grey #dad7d9

    percent_demonetized = []

    for category in VideoCategory.query.all():
        all_videos = make_integer(db.session.query(func.count(Video.video_id)
                              ).join(VideoCategory
                              ).filter(VideoCategory.video_category_id
                                       == category.video_category_id
                              ).first())

        demonetized_videos = make_integer(
                                db.session.query(func.count(Video.video_id)
                                      ).join(VideoCategory
                                      ).filter(VideoCategory.video_category_id
                                               == category.video_category_id
                                      ).filter(Video.is_monetized == False
                                      ).first())

        percent_demonetized.append((category.category_name.lower(), 
                                    round(demonetized_videos/all_videos*100)))

    percent_demonetized.sort(key=lambda x: (-x[1], x[0]))
    category_names = list(map(lambda x: x[0], percent_demonetized))
    percent_demonetized = list(map(lambda x: x[1], percent_demonetized))

    data = {'labels': category_names,
            'datasets': [{'label': '% demonetized',
                          'backgroundColor': colors,
                          'data': percent_demonetized}]}

    return jsonify(data)


@app.route('/explore/categories/<int:video_category_id>')
def show_specific_category_page(video_category_id):

    category = VideoCategory.query.get(video_category_id)

    videos_in_db = make_integer(
                        db.session.query(func.count(Video.video_id)
                                 ).join(VideoCategory
                                 ).filter(VideoCategory.video_category_id == video_category_id
                                 ).first())

    videos = RandomBag()

    for video in Video.query.filter(Video.video_category_id == video_category_id).all():
        videos.insert(video)
    
    random_videos = videos.get_random_elements(16)

    return render_template('category.html',
                            category=category,
                            videos_in_db=videos_in_db,
                            random_videos=random_videos)


@app.route('/explore/tags/')
def show_tags_page():

    tags = RandomBag()

    for tag in Tag.query.all():
        tags.insert(tag)
    
    random_tags = tags.get_random_elements(80)

    return render_template('tags.html',
                            tags=random_tags)

@app.route('/explore/tags2/')
def show_tags_page2():

    tags = RandomBag()

    for tag in Tag.query.all():
        tags.insert(tag)
    
    random_tags = tags.get_random_elements(80)

    return render_template('tags2.html',
                            tags=random_tags)


@app.route('/explore/tags/<int:tag_id>')
def show_specific_tag_page(tag_id):

    tag = Tag.query.get(tag_id)
    tag_videos = Video.query.join(TagVideo
                           ).filter(TagVideo.tag_id == tag_id
                           ).all()

    if len(tag_videos) < 5:
        random_videos = tag_videos
    else:
        tag_video_bag = RandomBag()
        for tag_video in tag_videos:
            tag_video_bag.insert(tag_video)

        random_videos = tag_video_bag.get_random_elements(4)

    return render_template('tag.html',
                            tag=tag,
                            videos_in_db=len(tag_videos),
                            random_videos=random_videos)


@app.route('/explore-tags-data.json')
def generate_tag_chart_data():

    all_tags = request.args.get('allTags')

    for tag in Tag.query.join(TagVideo).all():
        all_videos = make_integer(db.session.query(func.count(Video.video_id)
                              ).join(VideoCategory
                              ).filter(VideoCategory.video_category_id
                                       == category.video_category_id
                              ).first())
        demonetized_videos = make_integer(
                                db.session.query(func.count(Video.video_id)
                                      ).join(VideoCategory
                                      ).filter(VideoCategory.video_category_id
                                               == category.video_category_id
                                      ).filter(Video.is_monetized == False
                                      ).first())
        percent_demonetized.append((category.category_name.lower(), 
                                    round(demonetized_videos/all_videos*100)))

    percent_demonetized.sort(key=lambda x: (-x[1], x[0]))
    category_names = list(map(lambda x: x[0], percent_demonetized))
    percent_demonetized = list(map(lambda x: x[1], percent_demonetized))

    data = {'labels': category_names,
            'datasets': [{'label': '% demonetized',
                          'backgroundColor': colors,
                          'data': percent_demonetized}]}

    return jsonify(data)


# Error-related routes
@app.errorhandler(404)
def page_not_found(error):
    return render_template('errors/404.html')


@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html')


##### JSON routes


def process_period_tag_query(tag):

    period = [('2017-01-01', '2017-03-31'),
              ('2017-04-01', '2017-06-30'),
              ('2017-07-01', '2017-10-31'),
              ('2017-10-01', '2017-12-31'),
              ('2018-01-01', '2018-03-31'),
              ('2018-04-01', '2018-06-30')]

    # # A Flask-SQLAlchemy BaseQuery object for items in the tags_videos association table
    # basequery = db.session.query(TagVideo).join(Tag).filter(Tag.tag == tag)
    # tag_data_by_period = []

    # for quarter in range(len(period)): # use range() because we need to index into period
    #     # Filter basequery for each period
    #     basequery = basequery.filter(Videosideo.published_at >= period[quarter][0],
    #                                  Video.published_at < period[quarter][1])
        
    #     if basequery.count(): # if total videos with that tag is not zero
    #         demonetized_count = basequery.filter(Video.is_monetized == False).count()
    #         tag_data_by_period.append((round(demonetized_count/total_count*100), total_count))
    #         # ^^ numerator is the # of demonetized videos, denominator is total videos
            
    #     else: # if there are no videos with that tag for the specified period
    #         tag_data_by_period.append((0, 0))               
    
    # return tag_data_by_period

# - - - -
    # bqq = db.session.query(TagVideo).join(Tag).filter(Tag.tag == tag)
    # tag_period_data = []
    # for quarter in range(len(period)):
    #     demonetized_vids = db.session.query(TagVideo).join(Tag).filter(Tag.tag.like('%a')).join(Video).filter(Video.published_at >= period[quarter][0], Video.published_at < period[quarter][1]).filter(Video.is_monetized == True).count()
    #     all_vids = db.session.query(TagVideo).join(Tag).filter(Tag.tag.like('%a')).join(Video).filter(Video.published_at >= period[quarter][0], Video.published_at < period[quarter][1]).count()
    #     if all_vids:
    #         tag_period_data.append((round(demonetized_vids/all_vids*100), all_vids))
    #     else:  # if there are no videos you don't want a ZeroDivisionError
    #         tag_period_data.append((0, 0))
    # return tag_period_data

    bqq = db.session.query(TagVideo).join(Tag).filter(Tag.tag == tag)
    tag_period_data = []

    for quarter in range(len(period)):
        demonetized_vids = db.session.query(TagVideo
                                    ).join(Tag).filter(Tag.tag == tag
                                    ).join(Video
                                    ).filter(Video.published_at >= period[quarter][0], Video.published_at < period[quarter][1]
                                    ).filter(Video.is_monetized == False
                                    ).count()
        all_vids = db.session.query(TagVideo).join(Tag).filter(Tag.tag == tag).join(Video).filter(Video.published_at >= period[quarter][0], Video.published_at < period[quarter][1]).count()
        if all_vids:
            tag_period_data.append(round(demonetized_vids/all_vids*100))
        else:  # if there are no videos you don't want a ZeroDivisionError
            tag_period_data.append(0)
    return tag_period_data


@app.route('/initial-tag-data.json')
def create_initial_tag_data_json():
    """Query the database the retrieve tag demonetization data for example
    tags (to demonstrate chart's use) and return a json string.
    """

    colors = ['rgba(238, 39, 97, 1)', # pink
              'rgba(40, 37, 98, 1)', # purple
              'rgba(50, 178, 89, 1)', # green
              'rgba(94, 200, 213, 1)', # blue
              'rgba(255, 242, 0, 1)'] # yellow

    initial_tags = ['jake paul', 
                    'donald trump', 
                    'philip defranco']

    data = {'labels': ['q1_2017', # this is what we're passing to the front end
                       'q2_2017', 
                       'q3_2017', 
                       'q4_2017', 
                       'q1_2018',
                       'q2_2018'],
            'datasets': []
            }

    for i in range(len(initial_tags)):
        data_to_add = {'type': 'line',
                       'fill': False}
        data_to_add['label'] = initial_tags[i]
        data_to_add['borderColor'] = colors[i]
        data_to_add['data'] = process_period_tag_query(initial_tags[i])
        data['datasets'].append(data_to_add)

    return jsonify(data)


# @app.route('/get-tag-data.json')
# def create_tag_data_json():
#     """Query the database the retrieve relevant tag demonetization data and
#     return a json string."""

#     colors = ['rgba(238, 39, 97, 1)', # pink
#               'rgba(40, 37, 98, 1)', # purple
#               'rgba(50, 178, 89, 1)', # green
#               'rgba(94, 200, 213, 1)', # blue
#               'rgba(255, 242, 0, 1)'] # yellow

#     tag = request.args.get('tag-search-box')

#     data_to_add = {'type': 'line',
#                    'fill': False}
#     # Add necessary Chart.js datapoints
#     data_to_add['label'] = tag
#     data_to_add['borderColor'] = random.sample(colors, 1)
#     data_to_add['data'] = process_period_tag_query(tag)

#     data = {'labels': ['q1_2017', # this is what we're passing to the front end
#                        'q2_2017', 
#                        'q3_2017', 
#                        'q4_2017', 
#                        'q1_2018',
#                        'q2_2018'],
#             'datasets': []
#             }
            
#     return jsonify(data)


def count_tag_frequency(tag_item):
    """Count how many times a tag_related_item (tag_id or tag) appears in the 
    tags_videos association table."""

    # make things faster thru this? https://gist.github.com/hest/8798884
    tag_frequencies = []
    for item in tag_item:
        if type(item.tag) == int:
            tag_freq = db.session.query('*').filter(TagVideo.tag_id == item.tag_id).count()
            tag_frequencies.append(tag_freq)
        else:
            tag_freq = db.session.query(TagVideo).join(Tag).filter(Tag.tag == item.tag).count()
            tag_frequencies.append(tag_freq)

    return tag_frequencies


def trie_to_dict(node):
    """Return a trie in the form of a nested dictionary to be jsonified.
    
    >>> tag_trie = Trie()
    >>> trie_to_dict(tag_trie.root)
    {'': {'freq': 0, 'children': {}}}

    >>> tag_trie.add_word('a', 1)
    >>> trie_to_dict(tag_trie.root)
    {'': {'freq': 0, 'children': {'a': {'freq': 1, 'children': {}}}}}   
    
    >>> tag_trie.add_word('b', 1)
    >>> trie_to_dict(tag_trie.root)
    {'': {'freq': 0, 'children': {'a': {'freq': 1, 'children': {}}, 'b': {'freq': 1, 'children': {}}}}}   

    >>> tag_trie.add_word('an', 2)
    >>> trie_to_dict(tag_trie.root)
    {'': {'freq': 0, 'children': {'a': {'freq': 1, 'children': {'n': {'freq': 2, 'children': {}}}}, 'b': {'freq': 1, 'children': {}}}}}   

    >>> tag_trie.add_word('and', 3)
    >>> tag_trie.add_word('be', 2)
    >>> tag_trie.add_word('bee', 3)
    >>> tag_trie.add_word('being', 5)
    >>> trie_to_dict(tag_trie.root)
    {'': {'freq': 0, 'children': {'a': {'freq': 1, 'children': {'n': {'freq': 2, 'children': {}}}}, 'b': {'freq': 1, 'children': {}}}}}   
    """

    if not node.children: # leaf node (no children)
        return {'freq': node.freq, 'children': {} }

    trie_dict = {}
    for node_char in node.children:
        # Entering recursion for node with child(ren)
        trie_dict[node_char] = trie_to_dict(node.children[node_char])

    if not node.value: # root node scenario
        return {'': {'freq': 0, 'children': trie_dict}}

    # Q: Is it more pythonic to use else or just start with the return statement?
    # print('non-leaf-node recursion')
    return {'freq': node.freq, 'children': trie_dict}


def get_tag_frequency(tag_id):
    """Get tag frequency for a certain word from the db."""

    frequency_in_videos = db.session.query(func.count(TagVideo.tag_video_id)
                           ).filter(TagVideo.tag_id == tag_id
                           ).first()
    frequency_in_images = db.session.query(func.count(TagImage.tag_image_id)
                            ).filter(TagImage.tag_id == tag_id
                            ).first()

    return make_integer(frequency_in_videos) + make_integer(frequency_in_images)

    # by definition, there shouldn't be any tags whose frequency is zero.


def construct_tag_trie():
    """Construct a complete tag trie for all tags in the seed database (only needs 
    to be done once). Construct a smaller list of qualified tags and convert it into
    a dictionary for easy retrieval."""

    complete_tag_trie = Trie()
    smaller_tag_trie = Trie()

    tags = Tag.query.all()

    not_added = set()
    i = 0
    for tag in tags:
        if i%100 == 0:
            print('done processing {} out of {} tags'.format(i, len(tags)))
        tag_freq = get_tag_frequency(tag.tag_id)
        complete_tag_trie.add_word(tag.tag, tag_freq)
        if len(str(tag.tag)) < 18 and str(tag.tag).count(' ') < 4 and tag_freq > 3:
            smaller_tag_trie.add_word(str(tag.tag), tag_freq)
        else:
            # print('{} not added'.format(str(tag.tag)))
            not_added.add((str(tag.tag), tag_freq))
        i += 1

    tag_trie_dict = trie_to_dict(smaller_tag_trie.root)
    complete_tag_trie_dict = trie_to_dict(complete_tag_trie.root)

    sys.setrecursionlimit(30000) # exceed the default max recursion limit
    with open('tag_trie_dict2.pickle', 'wb') as f:
        pickle.dump(tag_trie_dict, f)

    with open('complete_tag_trie2.pickle', 'wb') as f:
        pickle.dump(complete_tag_trie, f)

    return [tag_trie_dict, complete_tag_trie_dict, not_added]


@app.route('/autocomplete-trie.json')
def return_tag_trie():
    """Return a jsonified dictionary representation of a trie for all qualified
    tags in the database."""
    with open('tag_trie_dict2.pickle', 'rb') as f:
        tag_trie_dict = pickle.load(f)

    return jsonify(tag_trie_dict)


def create_tag_list(trie_dict, previous=None):
    """
    Assume trie_dict is a dictionary representation of the tag trie.
    Return a list of all possible tags and their frequencies.

    >>> a_dict = {'a': {'freq': 1, 'children': {}}}
    >>> create_tag_list(a_dict, '')
    [['a', 1]]

    >>> a_be_dict = {'a': {'freq':1,
                           'children':{}},
                     'b':{'freq':0,
                          'children':{'e':{'freq': 2, 
                                           'children': {}}}}}
    >>> create_tag_list(a_be_dict, '')
    [['a', 1], ['be', 2]]

    >>> multi_word_dict = {'a': {'freq': 1,
                                 'children': {'n': {'freq': 2,
                                                    'children': {}}}},
                           'b': {'freq': 0,
                                 'children': {'e': {'freq': 2,
                                                    'children': {'t': {'children': {}, 'freq': 3}}}}}}
    >>> create_tag_list(multi_word_dict, '')
    [['a', 1], ['an', 2], ['be', 2], ['bet', 3]]

    >>> multi_level_dict = {'a':{'freq': 1,
                         'children':{'n':{'freq': 2,
                                          'children':{'d':{'children':{},'freq':3}}}}},
                    'b':{'freq':0,
                         'children':{'e':{'freq':1,
                                          'children':{'e':{'freq':3,
                                                           'children':{}},
                                                      'i':{'freq':0,
                                                           'children':{'n':{'freq':0,
                                                                            'children':{'g':{'freq':5,
                                                                                             'children':{}}}}},
                                                      't':{'freq':3,
                                                           'children':{}}}}}},
                                     'o':{'freq':0,
                                          'children':{'g':{'freq':3,
                                                            'children':{}},
                                                      'n':{'freq':0,
                                                            'children':{'g':{'freq':4,
                                                                             'children':{}}}}}}}}
    >>> create_tag_list(multi_level_dict, '')
    [['a',1],['an',2],['and',3],['be',2],['bee',3],['bet',3],['being',5],['bog',3],['bong',4]]
    """ 
    all_tags = []

    for char, info in trie_dict.items():  
        if not previous:
            previous = ''
            if info['freq']:
                all_tags.append((char, info['freq']))
            for key, value in info['children'].items():
                for item in create_tag_list({key: value}, char):
                    all_tags.append(item)
        else:
            if info['freq']:
                all_tags.append((previous+char, info['freq']))
            for key, value in info['children'].items():
                for item in create_tag_list({key: value}, (previous+char)):
                    all_tags.append(item)

    return all_tags


def get_all_words_from_prefix(trie_dict, prefix):
    """Given a prefix, find all valid tags that begin with it."""

    all_tags = []




def sort_tags_by_frequency(all_tags):
    """Assume all_tags is a list of all tags and their frequencies.
    Sort tags by frequency, then by tag length."""

    all_tags.sort(key=lambda x: (-x[1], x[0]))

    return all_tags


##################################
# from /add-data
##################################

@app.route('/add-data', methods=['GET', 'POST'])
def add_data():
    """Allow users to contribute additional creator data."""

    if request.method == 'POST':

        video_id = request.form['video-id-input']
        video = Video.query.filter(Video.video_id == video_id).first()

        monetization_status = request.form['monetizedRadio']
        if monetization_status == 'demonetized':
            is_monetized = False
        else:
            is_monetized = True

        added_on = datetime.datetime.utcnow()

        if video: # if the video is already in the db
            video.is_monetized = is_monetized
            addition = Addition(video_id=video_id,
                                added_on=added_on,
                                is_update=True)
            db.session.add(addition)
            db.session.commit()
            flash(Markup('''<div class="alert alert-success alert-dismissible" role="alert">
                                <button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>
                                Successfully updated the video's monetization status. <a href="/explore/videos/''' + video_id + '''" class="alert-link">Check it out</a> or add another.
                            </div>'''))
        
            return redirect('/add-data')

        else:
            addition = Addition(video_id=video_id,
                                added_on=added_on,
                                is_update=False)
            db.session.add(addition)
            video = Video(video_id=video_id,
                          is_monetized=is_monetized)
            db.session.add(video)
            db.session.commit()
            flash(Markup('''<div class="alert alert-success alert-dismissible" role="alert">
                                <button type="button" class="close" data-dismiss="alert" aria-label="Close"><span aria-hidden="true">&times;</span></button>
                                Successfully added the video. <a href="/explore/videos/''' + video_id + '''" class="alert-link">Check it out</a> or add another.
                            </div>'''))
            
            # call APIs to get other info
            add_all_info_to_db(video_id)

            return redirect('/add-data')

    else:
        return render_template('add-data.html')


@app.route('/check-database.json')
def check_user_submission():
    """Check the database to see if the user-submitted video is already in the
    database, and if so, whether the monetization status is the same.
    """

    video_id = request.args.get('videoId')
    monetized = request.args.get('monetizationStatus') #boolean
    
    if Video.query.filter(Video.video_id == video_id).first(): # if it exists
        if Video.query.filter(Video.is_monetized == True):
            database_status = 1
        else:
            database_status = 2
    else:
        database_status = 0
        # flash message directly?

    data = {'status': 
            database_status} # 0 if new video, 1 if existing + same monetization status, 2 if existing and different monetization status

    return jsonify(data)


@app.route('/change-ad-status.json')
def change_ad_status_in_db():
    """Update the video's monetization status in the database based on user feedback."""

    video_id = request.form['video-id-input']

    video = Video.query.filter(Video.video_id == video_id).first()
    if video.is_monetized == False:
        video.is_monetized = True
    else:
        video.is_monetized = False

    db.session.commit()
    flash("The video's ad status was successfully updated.") # tk add link


# - - - - - - - - - by channel size - - - - - - - - - -
# chartjs_default_data = {}

# @app.route('by-channel-size.json')
# def json_by_channel_size():
#     """Return
#     """
#     basequery = db.session.query(func.count(Video.video_id)).join(
#                     Channel).join(ChannelStat)

#     fully_monetized_vids = []
#     partially_monetized_vids = []
#     demonetized_vids = []
#     percent_data = []

#     data = [fully_monetized_vids, 
#             partially_monetized_vids, 
#             demonetized_vids]

#     for item in data:
#         basequery = basequery.filter(video.is_monetized == get_is_monetized(str(item))
#         elif item == partially_monetized_vids:
#             basequery = basequery.filter(video.is_monetized == True)

#         for tier in range(5):
#             if item == 0:
#                 is_monetized = 4
#             elif item == 1:
#                 is_monetized = 3
#             elif item == 2:
#                 is_monetized = 2
#             data[tier].append(basequery.filter(ChannelStat.total_subscribers >= 1000000,
#                                             Video.is_monetized == is_monetized).first())
#             data[0].append(basequery.filter(ChannelStat.total_subscribers >= 1000000,
#                                             Video.is_monetized == is_monetized).first())
#             data[0].append(basequery.filter(ChannelStat.total_subscribers >= 1000000,
#                                             Video.is_monetized == is_monetized).first())
#             data[0].append(basequery.filter(ChannelStat.total_subscribers >= 1000000,
#                                             Video.is_monetized == is_monetized).first())
#             data[0].append(basequery.filter(ChannelStat.total_subscribers >= 1000000,
#                                             Video.is_monetized == is_monetized).first())



#     tier1 = ChannelStat.total_subscribers >= 1000000
#     tier2 = ChannelStat.total_subscribers < 1000000,
#             ChannelStat.total_subscribers >= 500000
#     tier3 = ChannelStat.total_subscribers < 500000,
#             ChannelStat.total_subscribers >= 100000
#     tier4 = ChannelStat.total_subscribers < 100000,
#             ChannelStat.total_subscribers >= 500000
#     tier5 = ChannelStat.total_subscribers < 50000

#     tiers = [tier1, tier2, tier3, tier4, tier5]
#     for tier in tiers:
#       # Multiple base queries >> tack on filters as necessary.


# @app.route('/bleh-by-channel-size.json')
# def json_data_by_channel_size():
    """Return demonetization data by channel size."""

    # tier1_all_vids = db.session.query(
    #                     func.count(Video.video_id)).join(
    #                     ChannelStat).filter(
    #                     ChannelStat.total_subscribers > 1000000).first()
  
# @app.route('/autocomplete.json')
# def autocomplete_search():
#     tag = request.args.get('tagInput')
#     try:
#         tag = tag.lower().strip()
#     except AttributeError: # if tags are non-alphabet
#         print('AttributeError: ' + tag)
#     finally:
#         tag_search = Tag.query.filter(Tag.tag.like(str(tag) + '%')).all() # tag_search is a list of Tag objects
#         data = dict(zip(map(lambda x: x.tag, tag_search), count_tag_frequency(tag_search)))
#         print(data)
#         return jsonify(data)


if __name__ == '__main__':
    app.debug = True
    app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False
    app.jinja_env.auto_reload = app.debug
    connect_to_db(app)
    # DebugToolbarExtension(app)
    app.run(host='0.0.0.0')