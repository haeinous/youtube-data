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

# Load static tables

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

def load_ad_status(ad_status_filename):
    f = open(ad_status_filename)

    for row in f:
        row = row.rstrip()
        ad_status_id, ad_status_name = row.split(',')
        ad_status = AdStatus(ad_status_id=ad_status_id,
                             ad_status_name=ad_status_name)
        db.session.add(ad_status)

    db.session.commit()
    f.close()


def load_live_broadcast(live_broadcast_filename):
    f = open(live_broadcast_filename)

    for row in f:
        row = row.rstrip()
        live_broadcast_id, broadcast_status_name = row.split(',')
        live_broadcast = LiveBroadcast(live_broadcast_id=live_broadcast_id,
                                       broadcast_status_name=broadcast_status_name)
        db.session.add(live_broadcast)

    db.session.commit()
    f.close()


def load_textfield(textfield_filename):
    f = open(textfield_filename)

    for row in f:
        row = row.rstrip()
        textfield_id, textfield_name = row.split(',')
        textfield = Textfield(textfield_id=textfield_id,
                              textfield_name=textfield_name)
        db.session.add(textfield)

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


def load_language(language_filename):
    f = open(language_filename)

    for row in f:
        row = row.rstrip()
        language_code, language_name = row.split(',')
        language = Language(language_code=language_code,
                            language_name=language_name)


# Load data on video monetization status (raw data)

def load_video(video_filename):
    f = open(video_filename)

    for row in f:
        row = row.rstrip()
        video_id, ad_status_id = row.split(',')
        video = Video(video_id=video_id,
                      ad_status_id=ad_status_id)
        db.session.add(video)

    db.session.commit()
    f.close()


# Supplement raw data with YouTube API data

def populate_channel_data():
    """Populate the Channel table for videos."""
    # i = 0
    # while i < 15:
    # Get rid of the list brackets [ ] if / when we fetch all.
    blank_channel_ids = [Video.query.filter(Video.channel_id == None).first()]
    for item in blank_channel_ids:
        channel_id = get_channel_id_from_video_id(item.video_id)
        if not Channel.query.filter(Channel.channel_id == channel_id).first():
            # The above query will return None if the channel is not in the db.
            parse_channel_data(yt_info_by_id(channel_id))
            print('Commited a new channel: ' + channel_id)
        else:
            item.channel_id = channel_id                
            db.session.commit()
            print('Updated ' + item.video_id)
        # i += 1


def populate_video_data():
    """Populate the Video table."""

    blank_video_ids = Video.query.filter(Video.video_title.is_(None), 
                                         Video.channel_id.isnot(None)).all()
    for item in blank_video_ids:
        parse_video_data(yt_info_by_id(item.video_id))


def make_thumbnail_urls(video_id):
    """Assume video_id is a string.
    Return a string that is the thumbnail url."""

    return 'https://i.ytimg.com/vi/' + video_id + '/sddefault.jpg'


def populate_image_data():
    """Populate the ImageAnalysis table for videos with thumbnails."""

    # thumbnails = db.session.query(Video.video_id).filter(ImageAnalysis.video_id == None).all()
    video_ids_for_thumbnails = Video.query.filter(Video.video_id.isnot(None),
                                                  Video.channel_id.isnot(None)).all()
    image_urls = []
    for item in video_ids_for_thumbnails[:4]:
        print(item)
        if not ImageAnalysis.query.filter(ImageAnalysis.video_id == item.video_id).first():
            print('not')
            image_urls.append(make_thumbnail_urls(item.video_id))
            print(image_urls)
            process_images(image_urls)
        else:
            image_info = process_images(image_urls)
            for video_id in image_info:
                if 
            if TagImage.query.filter(TagImage.)

    # if len(thumbnail_list) > 128:
    #     thumbnails_to_process = thumbnail_list[:128]
    #     process_images(thumbnails_to_process)
    #     thumbnail_list = thumbnail_list[128:]
    # else:
    #     process_images(thumbnail_list)


def populate_text_data():
    """Populate the TextAnalysis table for titles, tags, and descriptions."""

    titles = db.session.query(Video.video_id).filter(TextAnalysis.text_field_id)
    pass



if __name__ == '__main__':
    connect_to_db(app)

    # video_category_filename = 'seed_data/video_category.csv'
    # ad_status_filename = 'seed_data/ad_status.csv'
    # textfield_filename = 'seed_data/textfield.csv'
    # country_filename = 'seed_data/country.csv'
    # live_broadcast_filename = 'seed_data/live_broadcast.csv'
    # language_filename = 'seed_data/language.csv'
    # video_filename = 'seed_data/video.csv'
    # load_video_category(video_category_filename)
    # load_ad_status(ad_status_filename)
    # load_live_broadcast(live_broadcast_filename)
    # load_textfield(textfield_filename)
    # load_country(country_filename)
    # load_video(video_filename)
    # load_language(language_filename)

    # populate_channel_data()
    # populate_video_data()
    # populate_image_data()
    # populate_text_data()
