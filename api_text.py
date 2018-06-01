#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""
import os, sys, requests, httplib2, statistics, difflib

from model import connect_to_db, db, TextAnalysis, Video, Tag, TagVideo, Channel, TagChannel

from googleapiclient import discovery
from googleapiclient.errors import HttpError

GOOGLE_KEY = os.environ.get('GOOGLE_KEY')
NLP_URL = 'https://language.googleapis.com/v1/documents:analyzeSentiment'


def call_nlp_api(text):

    discovery_url = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
    service = discovery.build('language',   
                              'v1',
                              http=httplib2.Http(),
                              discoveryServiceUrl=discovery_url,
                              developerKey=GOOGLE_KEY)
    service_request = service.documents().annotateText(
        body = {'document': 
                    {'type': 'PLAIN_TEXT',
                     'content': text},
                'features': 
                    {'extractDocumentSentiment': True},
                'encodingType': 'UTF8' if sys.maxunicode == 2047 else 'UTF32',
               }
        )

    try:
        nlp_response = service_request.execute()
    except HttpError as e:
        nlp_response = {'error': e}
        print(nlp_response)

    return nlp_response


def calculate_variation(sentences):

    scores = []

    for sentence in sentences:
        scores.append(sentence['sentiment']['score'])

    standard_deviation = round(statistics.stdev(scores),1)
    maximum = max(scores)
    minimum = min(scores)

    return [standard_deviation, maximum, minimum]


def add_to_db(nlp_response, youtube_id=None, textfield=None):
    """Assume nlp_response is dictionary of the JSON returned by the NLP API.
    Parse nlp_response and add data to the text_analyses table in db."""

    if 'error' in nlp_response:
        return

    else:
    sentiment_score = nlp_response['documentSentiment']['score']
    sentiment_magnitude = nlp_response['documentSentiment']['magnitude']
    language_code = nlp_response['language']
    if len(nlp_response['sentences']) > 1:
        standard_deviation, maximum, minimum = calculate_variation(nlp_response['sentences'])
    else:
        standard_deviation, maximum, minimum = None, None, None

    if len(youtube_id) == 11:
        video_id = youtube_id
        channel_id = None
    else:
        channel_id = youtube_id
        video_id = None

    text_analysis = TextAnalysis(video_id=video_id,
                                 channel_id=channel_id,
                                 textfield=textfield,
                                 sentiment_score=sentiment_score,
                                 sentiment_magnitude=sentiment_magnitude,
                                 sentiment_score_standard_deviation=standard_deviation,
                                 sentiment_max_score=maximum,
                                 sentiment_min_score=minimum,
                                 language_code=language_code)
    db.session.add(text_analysis)
    db.session.commit()


def has_meaningful_description(video_id):
    channel_id = get_channel_id_from_video_id(video_id)
    video_count = db.session.query(func.count(Video.video_id)).filter(Video.channel_id == 'UC_w1MuuO6WDhTtnGD-X1ZBQ').first()

    if make_int_from_sqa_object(video_count) > 5:
        videos = Video.query.filter(Video.channel_id == channel_id).limit(4).all()
        descriptions = list(map(lambda x: x.video_description, videos))
        a = descriptions[0]
        b = descriptions[1]
        c = descriptions[2]
        d = descriptions[3]

        diff_score1 = difflib.SequenceMatcher(None, a, b).quick_ratio()
        diff_score2 = difflib.SequenceMatcher(None, c, d).quick_ratio()

        if statistics.mean(diff_score1, diff_score2) < .5:
            return True

    return False   


def analyze_sentiment(youtube_id):
    """Call the Google NLP API, parse response, and add sentiment information 
    to the text_analyses table."""

    if len(youtube_id) == 11:
        # Can I do unpacking with this?
        video_title = Video.query.filter(Video.video_id == youtube_id).first().video_title
        add_to_db(call_nlp_api(video_title), youtube_id, 'video_title')
        video_description = Video.query.filter(Video.video_id == youtube_id).first().video_description
        add_to_db(call_nlp_api(video_description), youtube_id, 'video_description')

        video_tag_query = Tag.query.join(TagVideo).filter(TagVideo.video_id == youtube_id).all()
        if len(video_tag_query) > 5:
            video_tags = str([tag.tag for tag in video_tag_query])[1:-1]
            add_to_db(call_nlp_api(video_tags), youtube_id, 'video_tags')

    else:
        channel_description = Channel.query.filter(Channel.channel_id == youtube_id).first().channel_description
        add_to_db(call_nlp_api(channel_description), youtube_id, 'channel_description')
        channel_tag_query = Tag.query.join(TagChannel).filter(TagChannel.channel_id == youtube_id).all()
        
        if len(channel_tag_query) > 5:
            channel_tags = str([tag.tag for tag in channel_tag_query])[1:-1]
            add_to_db(call_nlp_api(channel_tags), youtube_id, 'channel_tags')


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()

