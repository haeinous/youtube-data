#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com
"""
from sqlalchemy import func

from server import app
from model import *

from api_youtube import *
from api_image import *
from api_text import *


# (1) Load static data tables establishee to maintain referential integrity
#     for video categories, live broadcast ids, and countries.

def load_video_category(video_category_filename):
    f = open(video_category_filename)

    for row in f:
        row = row.rstrip()
        video_category_id, category_name = row.split(',')
        video_category = VideoCategory(video_category_id=video_category_id,
                                       category_name=category_name)
        db.session.add(video_category)

    db.session.commit()
    f.close()


def load_country(country_filename):
    f = open(country_filename)

    for row in f:
        row = row.rstrip()
        # country_code, country_name, has_yt_space = row.split(',')
        country_list = row.split(',')
        country_code = country_list[0]
        country_name = country_list[1]
        has_yt_space = country_list[2]
        if has_yt_space == 'TRUE':
            has_yt_space = True
        else:
            has_yt_space = False
        country = Country(country_code=country_code,
                          country_name=country_name,
                          has_yt_space=has_yt_space)
        db.session.add(country)

    db.session.commit()
    f.close()

# (2.a) Load unique channels in seed data file (due to foreign key constraints,
#       channel data must be loaded before video data).
# (2.b) Populate the channels and channel_stats tables by calling the YouTube API.

def load_channel(channel_filename):
    """Load channel ids in the seed data."""

    with open(channel_filename) as f:
        for row in f:
            channel_id = row.rstrip()
            print(channel_id)
            add_channel_data(parse_channel_data(get_info_by_youtube_id(channel_id)))

        db.session.commit()


# (3.a) Load seed data on videos, monetization statuses, and channel IDs (raw seed data).

def load_video(video_filename):
    with open(video_filename) as f:
        for row in f:
            row = row.rstrip()
            video_id, is_monetized, channel_id = row.split(',')
            if is_monetized == 'TRUE':
                is_monetized = True
            else:
                is_monetized = False

            video = Video(video_id=video_id,
                          is_monetized=is_monetized,
                          channel_id=channel_id)
            db.session.add(video)

        db.session.commit()

# (3.b) Populate the videos and video_stats tables by calling the YouTube API.

def populate_video_data():
    """Populate the videos table."""

    # needs criterion for which videos to update

    all_videos = Video.query.all()
    for video in all_videos:
        video_id = video.video_id
        print(video_id)
        update_video_details(parse_video_data(get_info_by_youtube_id(video_id)))

# (4) Populate image_analyses data by calling the Clarifai API.

def populate_image_data():
    """Populate the image_analyses table for videos with thumbnails."""

    thumbnail_urls = [video.thumbnail_url for video in Video.query.all()]

    while len(thumbnail_urls) > 0:
        if len(thumbnail_urls) > 127:
            thumbnail_bunch = thumbnail_urls[:128]
            add_clarifai_data(thumbnail_urls)
            thumbnail_urls = thumbnail_urls[128:]
        else:
            add_clarifai_data(thumbnail_urls)

# (5) Populate text_analyses table with Google's NLP API.

def populate_text_data(youtube_id):
    """Populate the text_analyses table for titles, tags, and descriptions."""

    if meaningful_sentiment_likely(youtube_id):
        analyze_sentiment(youtube_id)


if __name__ == '__main__':
    connect_to_db(app)

    # video_category_filename = 'seed_data/video_category.csv'
    # country_filename = 'seed_data/country.csv'
    # load_video_category(video_category_filename)
    # load_country(country_filename)

    # channel_filename = 'seed_data/channel.csv'
    # load_channel(channel_filename)

    # video_filename = 'seed_data/video.csv'
    # load_video(video_filename)

    # populate_video_data()
    # populate_image_data()
    # populate_text_data()