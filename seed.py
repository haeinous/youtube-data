#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com
"""
from sqlalchemy import func
from sqlalchemy import exc

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

            try:
                db.session.commit()
            except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
                print(video_id + '\n' + str(e))
                db.session.rollback()

        print('Done loading seed video files!')

# (3.b) Populate the videos and video_stats tables by calling the YouTube API.

def populate_video_data():
    """Populate the videos table."""

    all_videos = Video.query.filter(Video.video_title.is_(None)
                           ).filter(Video.video_status.is_(None) # we want to dismiss deleted and errored-out videos
                           ).all()
    i = 0
    for video in all_videos:
        if i%100 == 0:
            print('done adding data for {} videos so far'.format(str(i)))
        video_id = video.video_id
        update_video_details(parse_video_data(get_info_by_youtube_id(video_id)))
        i += 1

# (4) Populate image_analyses data by calling the Clarifai API.

def populate_image_data():
    """Populate the image_analyses table for videos with thumbnails."""

    thumbnail_urls = [video.thumbnail_url for video in Video.query.all() if (video.video_title 
                                                                             and video.thumbnail_url 
                                                                             and not video.video_status)]
    loops = len(thumbnail_urls)//128+1
    print('loops: ' + str(loops))

    for i in range(1, loops+1):
        print('loop #' + str(i))
        if i == loops:
            add_clarifai_data(thumbnail_urls)
        else:
            add_clarifai_data(thumbnail_urls[:128])
            thumbnail_urls = thumbnail_urls[128:]
            

# (5) Populate text_analyses table with Google's NLP API.

def populate_text_data():
    """Populate the text_analyses table for titles, tags, and descriptions."""

    channel_ids = [channel.channel_id for channel in Channel.query.all() if len(TextAnalysis.query.filter(TextAnalysis.channel_id == channel.channel_id).all()) < 1]
    video_ids = [video.video_id for video in Video.query.all() if (len(TextAnalysis.query.filter(TextAnalysis.video_id == video.video_id
                                                                                        ).filter(Video.video_status.is_(None)
                                                                                        ).all()) < 3)]
    
    for channel_id in channel_ids:
        analyze_sentiment(channel_id)

    for video_id in video_ids:
        analyze_sentiment(video_id)


if __name__ == '__main__':
    connect_to_db(app)

    video_category_filename = 'seed_data/video_category.csv'
    country_filename = 'seed_data/country.csv'
    load_video_category(video_category_filename)
    load_country(country_filename)

    channel_filename = 'seed_data/channel.csv'
    load_channel(channel_filename)

    video_filename = 'seed_data/video.csv'
    load_video(video_filename)

    populate_video_data()
    populate_image_data()
    populate_text_data()