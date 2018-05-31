#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

What happens when a video and monetization status are submitted on the /add-data page.
"""

from model import *
from server import app
from api_youtube import *
from api_image import *
from api_text import *


def file_to_video_id_list(filename):

    with open(filename) as f:
        video_ids = []
        for line in f:
            line = line.strip()
            video_ids.append(line)
    return video_ids


def add_to_videos_table(filename):

    with open(filename) as f:
        video_ids = []
        for line in f:
            line = line.strip()
            video_id, is_monetized = line.split(',')
            video_ids.append(video_id)
            print(video_id, is_monetized)

            video = Video(video_id=video_id,
                          is_monetized=is_monetized)
            db.session.add(video)
        db.session.commit()

    add_all_info_to_db(video_ids)


def add_all_info_to_db(video_ids):
    """Given a list of video_ids, add all information to the database 
    in the correct order.
    """

    for video_id in video_ids:
        # Determine channel_id if necessary
        if not Video.query.filter(Video.video_id == video_id,
                              Video.channel_id.isnot(None)).first():
            channel_id = get_channel_id_from_video_id(video_id)
        else:
            channel_id = Video.query.filter(Video.video_id == video_id).first().channel_id
        
        # Add data to the channels and/or the channel_stats tables
        if Channel.query.filter(Channel.channel_id == channel_id).first():
            add_channel_stats_data(parse_channel_data(yt_info_by_id(channel_id), channel_in_db=True))
        else:
            add_channel_data(parse_channel_data(yt_info_by_id(channel_id)))

        # Add data to the video_stats and/or the videos, tags, and tags_videos tables
        if Video.query.filter(Video.video_id == video_id,
                              Video.video_title.isnot(None)).first():
            add_video_stats_data(parse_video_data(yt_info_by_id(video_id), video_details_in_db=True))
        else:
            add_video_details(parse_video_data(yt_info_by_id(video_id)))

        print('done adding YouTube data for {}!)'.format(video_id))

        # Add data to the image_analyses and text_analyses tables.
        if not ImageAnalysis.query.filter(ImageAnalysis.video_id == video_id).first():
            add_clarifai_data(video_ids)
            print('done adding image analysis data')

        # Add data for sentiment analysis
        if not TextAnalysis.query.filter(TextAnalysis.video_id == video_id).first():
            analyze_sentiment(video_ids)
            print('done adding text analysis data')


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()
