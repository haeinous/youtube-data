#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com
f
"""
import os, sys, requests, httplib2, statistics

from model import connect_to_db, db, TextAnalysis, Video, Tag, TagVideo, Channel, TagChannel
from sqlalchemy import exc
 
from googleapiclient import discovery
from googleapiclient.errors import HttpError

GOOGLE_KEY = os.environ.get('GOOGLE_KEY')
NLP_URL = 'https://language.googleapis.com/v1/documents:analyzeSentiment'

def call_nlp_api(text):
    """Assume text is a string from one of a video or channel's textfields.
    Call the NLP API and return the sentiment analysis."""

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
                'encodingType': 'UTF8' if sys.maxunicode == 2047 else 'UTF32'
                })

    try:
        nlp_response = service_request.execute()
    except HttpError as e:
        nlp_response = {'error': e}
        print(text, nlp_response)

    return nlp_response


def calculate_variation(sentences):
    """If the text field consists of multiple sentences, calculate standard 
    deviation as well as the maximum and minim for the sentiment scores."""
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
        if len(youtube_id) == 11:
            text_analysis = TextAnalysis(video_id=youtube_id,
                                         textfield='error')
        else:
            text_analysis = TextAnalysis(channel_id=youtube_id,
                                         textfield='error')
        db.session.add(text_analysis)
        try:
            db.session.commit()
        except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
            print(youtube_id + '\n' + str(e))
            db.session.rollback()
        
    sentiment_score = nlp_response['documentSentiment']['score']
    sentiment_magnitude = nlp_response['documentSentiment']['magnitude']
    if len(nlp_response['language']) < 5:
        language_code = nlp_response['language']
    else:
        language_code = nlp_response['language'][:4]

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

    try:
        db.session.commit()
    except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
        print(youtube_id + '\n' + str(e))
        db.session.rollback()


def analyze_sentiment(youtube_id):
    """Call the Google NLP API, parse response, and add sentiment information 
    to the text_analyses table."""

    if len(youtube_id) == 11:
        # Determine which videos need to their titles analyzed
        add_to_db(
            call_nlp_api(
                Video.query.filter(Video.video_id == youtube_id
                          ).first(
                          ).video_title), 
                youtube_id, 
                'video_title')

        # Determine which videos need their descriptions analyzed
        add_to_db(
            call_nlp_api(
                Video.query.filter(Video.video_id == youtube_id
                          ).first(
                          ).video_description), 
                youtube_id, 
                'video_description')

        # Determine which videos need their descriptions analyzed
        # video_tag_query = Tag.query.join(TagVideo).filter(TagVideo.video_id == youtube_id).all()

        # if len(video_tag_query) > 5:
        #     video_tags = str([tag.tag for tag in video_tag_query])[1:-1]
        #     add_to_db(call_nlp_api(video_tags), youtube_id, 'video_tags')

    else:
        # Analyze channel description
        add_to_db(
            call_nlp_api(
                Channel.query.filter(Channel.channel_id == youtube_id
                            ).first(
                            ).channel_description), 
                youtube_id, 
                'channel_description')

        # channel_tag_query = Tag.query.join(TagChannel).filter(TagChannel.channel_id == youtube_id).all()

        # if len(channel_tag_query) > 5:
        #     channel_tags = str([tag.tag for tag in channel_tag_query])[1:-1]
        #     add_to_db(call_nlp_api(channel_tags), youtube_id, 'channel_tags')


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()
