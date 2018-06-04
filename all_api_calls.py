#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

What happens when a video and monetization status are submitted on the /add-data page.
"""
from sqlalchemy import func

from model import *
# from server import app, make_int_from_sqa_object
from server import *
from api_youtube import *
from api_image import *
from api_text import *


def add_all_info_to_db(video_id):
    """Assume video_id is an 11-character string for a video that's not in the database.
    Add all relevant information to the database in the correct order."""

    # (1) Determine channel_id.
    channel_id = get_channel_id_from_video_id(video_id)
    
    # (2) Call the YouTube API to populate the channel table if it's not in the db.
    #     Otherwise, update the channel_stats table.
    if Channel.query.filter(Channel.channel_id == channel_id).first(): # returns None if channel not in db
        add_channel_stats_data(parse_channel_data(get_info_by_youtube_id(channel_id), channel_in_db=True))
    else: # if channel not in db:
        add_channel_data(parse_channel_data(get_info_by_youtube_id(channel_id)))

    # (3) Add data to the video and video_stats tables (tags and tags_videos if processing new tags).
    video = Video.query.filter(Video.video_id == video_id)
    update_video_details(parse_video_data(get_info_by_youtube_id(video_id)))
    # print('done adding YouTube data for {}!)'.format(video_id))

    # (4) Add data to the image_analyses table by calling the Clarifai API.
    thumbnail_url = Video.query.filter(Video.video_id == video_id).first().thumbnail_url
    add_clarifai_data(thumbnail_url)
    # print('done adding image analysis data')

    # (5) Add data to the text_analyses table if it's likely to yield interesting
    #     information by calling the Google NLP API for sentiment analysis. The free
    #     version of the API is capped, so not all videos should be sent over.
    if meaningful_sentiment_likely(video_id):
        analyze_sentiment(video_id)
    # print('done adding text analysis data')


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()