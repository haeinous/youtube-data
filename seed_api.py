#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com
"""

from model import *
from server import app
from api_youtube import *
from api_image import *
from api_text import *


def populate_channel_data():
    """Populate the Channel table for videos."""

    blank_channel_ids = Video.query.filter(Video.channel_id == None).first()
    print(blank_channel_ids)
    for video_id in blank_channel_ids:
        channel_id = get_channel_id_from_video_id(video_id)
        print(channel_id)
        parse_channel_data(yt_info_by_id(video_id))


def populate_video_data():
    """Populate the Video table."""

    blank_video_ids = Video.query.filter(Video.video_title == None).all()
    for video_id in blank_video_ids:
        video_id = video_id.video_id
        parse_video_data(yt_info_by_id(video_id))


def populate_image_data():
    """Populate the ImageAnalysis table for videos with thumbnails."""

    thumbnail_list = db.session.query(Video.video_id).filter(ImageAnalysis.video_id == None).all()

    if len(thumbnail_list) > 128:
        thumbnails_to_process = thumbnail_list[:128]
        process_images(thumbnails_to_process)
        thumbnail_list = thumbnail_list[128:]
    else:
        process_images(thumbnail_list)


def populate_text_data():
    """Populate the TextAnalysis table for titles, tags, and descriptions."""

    titles = db.session.query(Video.video_id).filter(TextAnalysis.text_field_id)
    pass
    


# if __name__ == '__main__':
#     connect_to_db(app)

    populate_channel_data()
    populate_video_data()
    populate_image_data()
    populate_text_data()