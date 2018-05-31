#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""
import os, sys, requests, json, datetime, httplib2, statistics

from model import connect_to_db, db, TextAnalysis, Video, Tag

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
                     'content': text,
                    },
                'features': 
                    {'extractDocumentSentiment': True
                    },
                'encodingType': 'UTF8' if sys.maxunicode == 65535 else 'UTF32',
               }
        )

    try:
        nlp_response = service_request.execute()
    except HttpError as e:
        nlp_response = {'error': e}
        print(nlp_response)

    return nlp_response


def calculate_sentiment_variation(sentences):

    # magnitudes = []
    scores = []

    for sentence in sentences:
        # magnitudes.append(sentence['sentiment']['magnitude'])
        scores.append(sentence['sentiment']['score'])

    if len(scores) > 1:
        standard_deviation = statistics.stdev(scores)
        maximum = max(scores)
        minimum = min(scores)
        return [standard_deviation, maximum, minimum]
    else:
        return [None, None, None]


def add_to_db(nlp_response, video_id, textfield_id):
    """Assume nlp_response is dictionary of the JSON returned by the NLP API.
    Parse nlp_response and add data to the text_analyses table in db."""

    sentiment_score = nlp_response['documentSentiment']['score']
    sentiment_magnitude = nlp_response['documentSentiment']['magnitude']
    language_code = nlp_response['language']
    standard_deviation, maximum, minimum = calculate_sentiment_variation(nlp_response['sentences'])
    print('stdev={}, max={}, min={}'.format(standard_deviation, maximum, minimum))

    text_analysis = TextAnalysis(video_id=video_id,
                                 textfield_id=textfield_id,
                                 sentiment_score=sentiment_score,
                                 sentiment_magnitude=sentiment_magnitude,
                                 # sentiment_score_standard_deviation=standard_deviation,
                                 # sentiment_max_score=maximum,
                                 # sentiment_min_score=minimum,
                                 language_code=language_code)
    db.session.add(text_analysis)

    db.session.commit()


def analyze_sentiment(video_ids):
    """Assume video_ids is a list of YouTube video ids.
    Call the Google NLP API, parse response, and add sentiment information 
    to the text_analyses table.
    """
    for video_id in video_ids:
        video_title = db.session.query(Video.video_title).filter(Video.video_id == video_id).first()
        video_description = db.session.query(Video.video_description).filter(Video.video_id == video_id).first()
        
        video_title = str(video_title)[1:-2]
        video_description = str(video_description)[1:-2]

        if video_title != 'None':
            print('video_id: {}, title: {}, description: {}'.format(video_id, video_title, video_description))

        video_title_analysis = call_nlp_api(video_title)
        # video_description_analysis = call_nlp_api(video_description)

        add_to_db(video_title_analysis, video_id, 1)
        # add_to_db(video_description_analysis, video_id, 2)

        # Do one for tags?


def test_this(filename):
    with open(filename) as f:
        video_ids = []
        for line in f:
            line = line.strip()
            video_ids.append(line)

    analyze_sentiment(video_ids)


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()






