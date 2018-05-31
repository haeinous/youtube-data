#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""

import datetime, random
from jinja2 import StrictUndefined
from flask import Flask, render_template, request, flash, redirect, session, jsonify
from flask_debugtoolbar import DebugToolbarExtension
from sqlalchemy import func

from model import *
from api_youtube import *
# from tries import Trie, TrieNode, TrieWord, create_prefix_trie

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
        current = self.root

        for char in word:
            if char in current.children:
                current = current.children[char]
            else:
                current.add_child(char)
                current = current.children[char]

        current.freq = freq
    
    def __repr__(self):
        return '<Trie root={}>'.format(self.root)


class TrieNode:

    def __init__(self, data):
        self.data = data
        self.children = dict() # key is char, value is TrieNode object
        self.freq = 0

    def add_child(self, data):
        child_node = TrieNode(data)
        self.children[data] = child_node

    def __repr__(self):
        return '<TrieNode data={}, freq={}, children={}>'.format(self.data,
                                                                 self.freq,
                                                                 self.children)


##### Helper functions

def stringify_sqa_object(object):
    """Assume object is a sqlalchemy.util._collections.result.
    Return a string."""

    return str(object)[1:-2]

def make_int_from_sqa_object(object):
    """Assume object is a sqlalchemy.util._collections.result.
    Return an int."""

    return int(str(object)[1:-2])


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

    videos_in_db = db.session.query(func.count(Video.video_id)).first()
    channels_in_db = db.session.query(func.count(Channel.channel_id)).first()
    videos_in_db = stringify_sqa_object(videos_in_db)
    channels_in_db = stringify_sqa_object(channels_in_db)

    # return render_template('homepage.html', form=search_form,
    #                                         videos_in_db=videos_in_db,
    #                                         channels_in_db=channels_in_db)
    return render_template('homepage.html', videos_in_db=videos_in_db,
                                            channels_in_db=channels_in_db)


@app.route('/chart_test', methods=['GET','POST'])
def chart():

    return render_template('chart_test.html')


@app.route('/chart-tag', methods=['GET', 'POST'])
def show_charts():

    return render_template('chart-tag.html')


@app.route('/search')
def display_search_results():
    """Display search results."""

    search_term = request.args.get('q')
    # search_term = 'something'

    search_results = [['Channels', ('UCe2ba_7LwQdNK5BKFSLm41Q', 'Tessa Brooks'), 
                                  ('UCzUV5283-l5c0oKRtyenj6Q', 'Mark Dice')],
                     ['Tags', (938, 'dobre twins'), 
                               (944, 'team ten')],
                     ['Categories', (10, 'Music')],
                     ['Videos', ('0EoGyT4f8Lc', 'Paris... The City Of Love ;) (Shameless Beauty Launch)'),
                                  ('vuA9G2RBwug', 'MARCH FASHION LOOKBOOK')]]

    if not search_results:
        flash('No search results!')
        return redirect('/')

    else:
        print(search_results)
        return render_template('search-results.html', search_results=search_results,
                                                      search_term=search_term)


@app.route('/about')
def about_page():
    """About page."""

    videos_in_db = db.session.query(func.count(Video.video_id)).first()
    channels_in_db = db.session.query(func.count(Channel.channel_id)).first()
    videos_in_db = stringify_sqa_object(videos_in_db)
    channels_in_db = stringify_sqa_object(channels_in_db)

    return render_template('about.html', videos_in_db=videos_in_db,
                                         channels_in_db=channels_in_db)


@app.route('/explore/', methods=['GET'])
def explore_page():

    return render_template('explore.html')


@app.route('/explore/channels/', methods=['GET'])
def show_channels_page():
    """Show a random selection of YouTube channels."""
    
    random_channels = random.sample(Channel.query.all(), 8)

    return render_template('channels.html',
                           channels=random_channels)



@app.route('/explore/channels/<channel_id>')
def show_specific_channel_page(channel_id):
    """Show info about a creator's videos."""

    channel = Channel.query.get(channel_id) # channel is a Channel object
    videos = Video.query.filter(
                    Video.channel_id == channel_id).order_by(
                    Video.published_at.desc()).all()
    videos_in_db = str(db.session.query(func.count(
                        Video.video_id)).filter(
                        Video.channel_id == channel_id).first())[1:-2]

    # Update the channel_stats table with the most up-to-date info
    add_channel_stats_data(parse_channel_data(yt_info_by_id(channel_id), channel_in_db=True))    
    channel_stats = ChannelStat.query.filter(ChannelStat.channel_id == channel_id).first()

    return render_template('channel.html',
                            channel=channel,
                            channel_stats=channel_stats,
                            videos=videos,
                            videos_in_db=videos_in_db)


@app.route('/explore/videos/')
def show_videos_page():
    """Show a random selection of YouTube videos."""

    random_videos = random.sample(Video.query.filter(
                        Video.channel_id.isnot(None)).filter(
                        Video.video_title.isnot(None)).all(), 8)

    return render_template('videos.html',
                            random_videos=random_videos)


@app.route('/explore/videos/<video_id>')
def show_specific_video_page(video_id):
    """Show info about a specific video."""

    video = Video.query.filter(Video.video_id == video_id).first()
    thumbnail_url = video.thumbnail_url

    # Update the video_stats table with the most up-to-date info
    add_video_stats_data(parse_video_data(yt_info_by_id(video_id)))    
    video_stats = VideoStat.query.filter(VideoStat.video_id == video_id).first()

    channel = Channel.query.join(Video).filter(Video.video_id == video_id).first()
    image_analysis = ImageAnalysis.query.filter(ImageAnalysis.video_id == video_id).first()
    if image_analysis:
        nsfw_score = round(image_analysis.nsfw_score * 100)
        colors = [color.color_hex_code for color in ColorImage.query.filter(ColorImage.image_analysis_id == image_analysis.image_analysis_id).all()]
    else:
        nsfw_score = None
        colors = []
    text_analyses = TextAnalysis.query.filter(TextAnalysis.video_id == video_id).all()
    text_analyses = [(text_analysis.textfield_id, 
                      text_analysis.sentiment_score, 
                      text_analysis.sentiment_magnitude) for text_analysis in text_analyses] # list comprehension
    if Tag.query.join(TagVideo).filter(TagVideo.video_id == video_id).first():
        tags = [tag.tag for tag in Tag.query.join(TagVideo).filter(TagVideo.video_id == video_id).all()]
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
                            colors=colors,
                            text_analyses=text_analyses,
                            tags=tags,
                            category=category,
                            duration=duration)


@app.route('/explore/categories/')
def show_categories_page():

    categories = VideoCategory.query.all()

    return render_template('categories.html',
                            categories=categories)


@app.route('/explore/categories/<int:video_category_id>')
def show_specific_category_page(video_category_id):

    category = VideoCategory.query.filter(VideoCategory.video_category_id == video_category_id).first()

    videos_in_db = str(db.session.query(func.count(
                        Video.video_id)).join(VideoCategory).filter(
                        VideoCategory.video_category_id == video_category_id).first())[1:-2]

    if int(videos_in_db) < 4:
        random_videos = Video.query.join(VideoCategory).filter(
                            VideoCategory.video_category_id == video_category_id).all()

    else:
        demonetized_videos = random.sample(Video.query.join(
                                VideoCategory).filter(
                                Video.is_monetized == False).filter(
                                VideoCategory.video_category_id == video_category_id).all(), 3)
        monetized_videos = random.sample(Video.query.join(
                                VideoCategory).filter(
                                    Video.is_monetized == True).filter(
                                    VideoCategory.video_category_id == video_category_id).all(), 3)
        random_videos = demonetized_videos + monetized_videos

    return render_template('category.html',
                            category=category,
                            videos_in_db=videos_in_db,
                            random_videos=random_videos)


@app.route('/explore/tags/')
def show_tags_page():

    desired_items_on_page = 80

    tags = random.sample(Tag.query.all(), desired_items_on_page)
    tags = sorted(tags, key=lambda x: x.tag)
    return render_template('tags.html',
                            tags=tags)


@app.route('/explore/tags/<int:tag_id>')
def show_specific_tag_page(tag_id):

    tag = Tag.query.filter(Tag.tag_id == tag_id).first()

    tag_videos = Video.query.join(TagVideo).filter(TagVideo.tag_id == tag_id).all()

    num_videos = len(tag_videos)

    if len(tag_videos) < 4:
        random_videos = random.shuffle(tag_videos, len(tag_videos))
    else:
        random_videos = random.sample(tag_videos, 4)

    return render_template('tag.html',
                            tag=tag,
                            videos_in_db=num_videos,
                            random_videos=random_videos)


@app.route('/add-data', methods=['GET', 'POST'])
def add_data():
    """Allow users to contribute additional creator data."""

    if request.method == 'POST':

        video_id = request.form['video-id-input']
        video = Video.query.filter(Video.video_id == video_id).first()

        monetization_status = request.form['monetizationStatus']
        print(monetization_status)
        if monetization_status == 'demonetized':
            is_monetized = False
        else:
            is_monetized = True

        submitted_time = datetime.datetime.utcnow()

        if video:
            video.is_monetized = is_monetized
            video.updated_at = submitted_time
            db.session.commit()
            flash("Successfully updated the video's monetization status!")
            return

        else:
            video = Video(video_id=video_id,
                          is_monetized=is_monetized,
                          submitted_time=submitted_time)
            db.session.add(video)
            db.session.commit()
            flash("Successfully added the video. Check it out or add another.") # tk add link
            return

    else:
        return render_template('add-data.html')

# Error-related routes

# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('errors/404.html')


# @app.errorhandler(500)
# def internal_error(error):
#     return render_template('errors/500.html') #tk need to create this


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
        demonetized_vids = db.session.query(TagVideo).join(Tag).filter(Tag.tag == tag).join(Video).filter(Video.published_at >= period[quarter][0], Video.published_at < period[quarter][1]).filter(Video.is_monetized == False).count()
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


@app.route('/get-tag-data.json')
def create_tag_data_json():
    """Query the database the retrieve relevant tag demonetization data and
    return a json string."""

    colors = ['rgba(238, 39, 97, 1)', # pink
              'rgba(40, 37, 98, 1)', # purple
              'rgba(50, 178, 89, 1)', # green
              'rgba(94, 200, 213, 1)', # blue
              'rgba(255, 242, 0, 1)'] # yellow

    tag = request.args.get('tag-search-box')

    data_to_add = {'type': 'line',
                   'fill': False}
    # Add necessary Chart.js datapoints
    data_to_add['label'] = tag
    data_to_add['borderColor'] = random.sample(colors, 1)
    data_to_add['data'] = process_period_tag_query(tag)

    data = {'labels': ['q1_2017', # this is what we're passing to the front end
                       'q2_2017', 
                       'q3_2017', 
                       'q4_2017', 
                       'q1_2018',
                       'q2_2018'],
            'datasets': []
            }
            
    return jsonify(data)


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
    print(data)
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


def count_tag_frequency(tag_item):
    """Assume Count how many times a tag_related_item (tag_id or tag) appears
    in the TagVideo table."""

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


@app.route('/autocomplete.json')
def autocomplete_search():
    tag = request.args.get('tagInput')
    try:
        tag = tag.lower().strip()
    except AttributeError: # if tags are non-alphabet
        print('AttributeError: ' + tag)
    finally:
        tag_search = Tag.query.filter(Tag.tag.like(str(tag) + '%')).all() # tag_search is a list of Tag objects
        data = dict(zip(map(lambda x: x.tag, tag_search), count_tag_frequency(tag_search)))
        print(data)
        return jsonify(data)


def trie_to_dict(node):
    """Assume node is the root node of a trie.
    Return a nested dictionary to be converted to JSON.
    
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
        return {'freq': node.freq,
                'children': {} }

    trie_dict = {}
    for node_char in node.children:
        # Entering recursion for node with child(ren)
        trie_dict[node_char] = trie_to_dict(node.children[node_char])

    if not node.data: # root node scenario
        return {'': {'freq': 0,
                     'children': trie_dict}}

    # Q: Is it more pythonic to use else or just start with the return statement?
    # print('non-leaf-node recursion')
    return {'freq': node.freq,
            'children': trie_dict}


def get_tag_frequency(word):
    """Get tag frequency for a certain word from the db."""

    frequency_in_videos = db.session.query(func.count(TagVideo.tag_video_id)
                            ).join(Tag
                            ).filter(Tag.tag == word
                            ).first()

    frequency_in_images = db.session.query(func.count(TagImage.tag_image_id)
                            ).join(Tag
                            ).filter(Tag.tag == word
                            ).first()

    return frequency_in_videos + frequency_in_images # need to make an integer from query object
    # There shouldn't be any tags whose frequency is zero.

@app.route('/autocomplete-trie.json')
def construct_tag_trie():
    """Return a jsonified dictionary representation of a trie for all the
    tags in the database."""

    trie = Trie()

    for tag in Tag.query.all():
        trie.add_word(tag.tag, get_tag_frequency(tag.tag))

    # turn trie into a dictionary so it can be jsonified
    trie_dict = trie_to_dict(trie.root)
    print(trie_dict)
    return jsonify(trie_dict)


@app.route('/check-database.json')
def check_database_for_duplicates():

    video_id = request.args.get('video-id-input')
    video = Video.query.filter(Video.video_id == video_id).first()
    video_info = {}

    if video:
        video_info['in_database'] = True
        video_info['is_monetized'] = video.is_monetized

    return jsonify(video_info)

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


@app.route('/bleh-by-channel-size.json')
def json_data_by_channel_size():
    """Return demonetization data by channel size."""

    # tier1_all_vids = db.session.query(
    #                     func.count(Video.video_id)).join(
    #                     ChannelStat).filter(
    #                     ChannelStat.total_subscribers > 1000000).first()
    tier1_all_vids = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(ChannelStat).filter(
                        ChannelStat.total_subscribers >= 1000000).first())
    tier1_fully_monetized = make_int_from_sqa_object(db.session.query(
                            func.count(Video.video_id)).join(
                            Channel).join(
                            ChannelStat).filter(
                            ChannelStat.total_subscribers >= 1000000,
                            Video.is_monetized == True).first())
    tier1_partially_monetized = make_int_from_sqa_object(db.session.query(
                                func.count(Video.video_id)).join(
                                Channel).join(
                                ChannelStat).filter(
                                ChannelStat.total_subscribers >= 1000000,
                                Video.is_monetized == True).first())
    tier1_demonetized = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(
                        ChannelStat).filter(
                        ChannelStat.total_subscribers >= 1000000,
                        Video.is_monetized == False).first())
    try:
        tier1_percent_demonetized = round(tier1_demonetized/tier1_all_vids, 2)
    except ZeroDivisionError:
        tier1_percent_demonetized = 0

###################

    tier2_all_vids = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(ChannelStat).filter(
                        ChannelStat.total_subscribers < 1000000,
                        ChannelStat.total_subscribers >= 500000).first())
    tier2_fully_monetized = make_int_from_sqa_object(db.session.query(
                            func.count(Video.video_id)).join(
                            Channel).join(
                            ChannelStat).filter(
                            ChannelStat.total_subscribers < 1000000,
                            ChannelStat.total_subscribers >= 500000,
                            Video.is_monetized == True).first())
    tier2_partially_monetized = make_int_from_sqa_object(db.session.query(
                                func.count(Video.video_id)).join(
                                Channel).join(
                                ChannelStat).filter(
                                ChannelStat.total_subscribers < 1000000,
                                ChannelStat.total_subscribers >= 500000,
                                Video.is_monetized == True).first())
    tier2_demonetized = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(
                        ChannelStat).filter(
                        ChannelStat.total_subscribers < 1000000,
                        ChannelStat.total_subscribers >= 500000,
                        Video.is_monetized == False).first())
    try:
        tier2_percent_demonetized = round(tier2_demonetized/tier2_all_vids, 2)
    except ZeroDivisionError:
        tier2_percent_demonetized = 0

###################    

    tier3_all_vids = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(ChannelStat).filter(
                        ChannelStat.total_subscribers < 500000,
                        ChannelStat.total_subscribers >= 100000).first())
    tier3_fully_monetized = make_int_from_sqa_object(db.session.query(
                            func.count(Video.video_id)).join(
                            Channel).join(
                            ChannelStat).filter(
                            ChannelStat.total_subscribers < 500000,
                            ChannelStat.total_subscribers >= 100000,
                            Video.is_monetized == True).first())
    tier3_partially_monetized = make_int_from_sqa_object(db.session.query(
                                func.count(Video.video_id)).join(
                                Channel).join(
                                ChannelStat).filter(
                                ChannelStat.total_subscribers < 500000,
                                ChannelStat.total_subscribers >= 100000,
                                Video.is_monetized == True).first())
    tier3_demonetized = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(
                        ChannelStat).filter(
                        ChannelStat.total_subscribers < 500000,
                        ChannelStat.total_subscribers >= 100000,
                        Video.is_monetized == False).first())
    try:
        tier3_percent_demonetized = round(tier3_demonetized/tier3_all_vids, 2)
    except ZeroDivisionError:
        tier3_percent_demonetized = 0   

    tier4_all_vids = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(ChannelStat).filter(
                        ChannelStat.total_subscribers < 100000,
                        ChannelStat.total_subscribers >= 50000).first())
    tier4_fully_monetized = make_int_from_sqa_object(db.session.query(
                            func.count(Video.video_id)).join(
                            Channel).join(
                            ChannelStat).filter(
                            ChannelStat.total_subscribers < 100000,
                            ChannelStat.total_subscribers >= 50000,
                            Video.is_monetized == True).first())
    tier4_partially_monetized = make_int_from_sqa_object(db.session.query(
                                func.count(Video.video_id)).join(
                                Channel).join(
                                ChannelStat).filter(
                                ChannelStat.total_subscribers < 100000,
                                ChannelStat.total_subscribers >= 50000,
                                Video.is_monetized == True).first())
    tier4_demonetized = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(
                        ChannelStat).filter(
                        ChannelStat.total_subscribers < 100000,
                        ChannelStat.total_subscribers >= 50000,
                        Video.is_monetized == False).first())
    try:
        tier4_percent_demonetized = round(tier4_demonetized/tier4_all_vids, 2)
    except ZeroDivisionError:
        tier4_percent_demonetized = 0

    tier5_all_vids = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(ChannelStat).filter(
                        ChannelStat.total_subscribers < 50000).first())
    tier5_fully_monetized = make_int_from_sqa_object(db.session.query(
                            func.count(Video.video_id)).join(
                            Channel).join(
                            ChannelStat).filter(
                            ChannelStat.total_subscribers < 50000,
                            Video.is_monetized == True).first())
    tier5_partially_monetized = make_int_from_sqa_object(db.session.query(
                                func.count(Video.video_id)).join(
                                Channel).join(
                                ChannelStat).filter(
                                ChannelStat.total_subscribers < 50000,
                                Video.is_monetized == True).first())
    tier5_demonetized = make_int_from_sqa_object(db.session.query(
                        func.count(Video.video_id)).join(
                        Channel).join(
                        ChannelStat).filter(
                        ChannelStat.total_subscribers < 50000,
                        Video.is_monetized == False).first())
    try:
        tier5_percent_demonetized = round(tier5_demonetized/tier5_all_vids, 2)
    except ZeroDivisionError:
        tier5_percent_demonetized = 0


    percent_data = [tier5_percent_demonetized,
                    tier4_percent_demonetized,
                    tier3_percent_demonetized,
                    tier2_percent_demonetized,
                    tier1_percent_demonetized]
    fully_monetized_data = [tier5_fully_monetized,
                            tier4_fully_monetized,
                            tier3_fully_monetized,
                            tier2_fully_monetized,
                            tier1_fully_monetized]
    partially_monetized_data = [tier5_partially_monetized,
                                tier4_partially_monetized,
                                tier3_partially_monetized,
                                tier2_partially_monetized,
                                tier1_partially_monetized]
    demonetized_data = [tier5_demonetized,
                        tier4_demonetized,
                        tier3_demonetized,
                        tier2_demonetized,
                        tier1_demonetized]


    data = {'labels': ['10k–50k',
                       '50k–100k',
                       '100k–500k',
                       '500k–1m',
                       '1m+'],
            'datasets': [{'type': 'line',
                          'label': '% Demonetized',
                          'borderColor': 'rgba(40,37,98,.8)',
                          'borderWidth': 4,
                          'fill': False,
                          'pointRadius': 8,
                          'data': percent_data,
                          'yAxisID': 'y-axis-2'
                         },
                         {'type': 'bar',
                          'label': 'Fully monetized',
                          'backgroundColor': 'rgba(50,178,89,.8)',
                          'borderWidth': 0,
                          'pointRadius': 8,
                          'data': fully_monetized_data,
                          'yAxisID': 'y-axis-1'
                         },
                         {'type': 'bar',
                          'label': 'Partially monetized',
                          'backgroundColor': 'rgba(94,200,213,.8)',
                          'borderWidth': 0,
                          'pointRadius': 8,
                          'data': partially_monetized_data,
                          'yAxisID': 'y-axis-1'
                         },
                         {'type': 'bar',
                          'label': 'Demonetized',
                          'backgroundColor': 'rgba(238,39,97,.8)',
                          'borderWidth': 0,
                          'pointRadius': 8,
                          'data': demonetized_data,
                          'yAxisID': 'y-axis-1'
                        }]
           }

    return jsonify(data)






if __name__ == '__main__':
    app.debug = True
    app.jinja_env.auto_reload = app.debug
    connect_to_db(app)
    DebugToolbarExtension(app)
    app.run(host='0.0.0.0')
