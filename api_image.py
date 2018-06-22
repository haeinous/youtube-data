#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""

import os, json
from pprint import pprint
from clarifai.rest import ClarifaiApp, ApiError
from clarifai.rest import Image as ClImage
from model import connect_to_db, db, ImageAnalysis, Tag, Color, TagImage, ColorImage, Video
from sqlalchemy import exc

CLARIFAI_KEY = os.environ.get('CLARIFAI_KEY')


####### new/single ########
# def add_clarifai_data(thumbnail_url):

#     image_info = parse_image_data(thumbnail_url)
#     if image_info:
#         add_non_duplicate_data(image_info)


def add_clarifai_data(thumbnail_urls):

    add_to_database(parse_image_data(thumbnail_urls))


####### old/bulk ########
# Create a list of initialized Clarifai images (class ClImage) and call the Clarifai API

def add_tag_data(tag):
    """Assume tag is a tag in string form.
    Add it to the database."""

    add_tag = Tag(tag=tag)
    db.session.add(add_tag)
    try:
        db.session.commit()
    except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
        print(tag + '\n' + str(e))


def add_color_data(hex_code, color_name):
    """Assume hex_code is a 7-character string that's a hex code.
    Add it to the database."""

    color = Color(hex_code=hex_code.rstrip().lower(),
                      color_name=color_name.rstrip().lower())
    db.session.add(color)
    try:
        db.session.commit()
    except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
        print(hex_code + '\n' + str(e))

def add_tag_image_data(video_id, info, image_analysis_id):
    """Assume image_info is a dictionary with video_ids as keys and
    image_analysis_id is the int primary key for the image_analyses table.
    Populate the tags_images association table."""

    tags = info['tags'] # tags is a dictionary
    for tag_to_add in tags:
        tag_id = Tag.query.filter(Tag.tag == tag_to_add).first().tag_id
        if not TagImage.query.filter(TagImage.tag_id == tag_id,
                                     TagImage.image_analysis_id == image_analysis_id).first():
            tag_image = TagImage(tag_id=tag_id, 
                                 image_analysis_id=image_analysis_id)
            db.session.add(tag_image)

    try:
        db.session.commit()
    except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
        print(hex_code + '\n' + str(e))


def add_color_image_data(video_id, info, image_analysis_id):
    """Assume image_info is a dictionary with video_ids as keys and
    image_analysis_id is the int primary key for the image_analyses table.
    Populate the colors_images association table."""

    colors = info['colors'] # colors is a dictionary
    for color_to_add in colors:
        hex_code = Color.query.filter(Color.hex_code == color_to_add).first().hex_code
        if not ColorImage.query.filter(ColorImage.hex_code == hex_code,
                                       ColorImage.image_analysis_id == image_analysis_id).first():        
            color_image = ColorImage(hex_code=hex_code,
                                     image_analysis_id=image_analysis_id)
            db.session.add(color_image)

    try:
        db.session.commit()
    except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
        print(hex_code + '\n' + str(e))


def add_nsfw_image_data(video_id, info):
    """Populate the image_analyses table."""

    nsfw_score = info['nsfw_score']
    image_analysis = ImageAnalysis(video_id=video_id,
                                   nsfw_score=nsfw_score)
    db.session.add(image_analysis)

    try:
        db.session.commit()
    except (Exception, exc.SQLAlchemyError, exc.InvalidRequestError, exc.IntegrityError) as e:
        print(hex_code + '\n' + str(e))


def parse_image_data(image_urls):
    """Assume image_urls is a list of image urls.
    Return a dictionary of tags and nsfw scores associated with each thumbnail image
    (video_ids are keys).
    """

    # Initialize images.
    initialized_ClImages = list(map(ClImage, image_urls))
    app = ClarifaiApp(api_key=CLARIFAI_KEY)
    image_info = {}

    # Obtain relevant tag information
    try:
        general_response = app.models.get('general-v1.3').predict(initialized_ClImages)
    except ApiError as e:
        error = json.loads(e.response.content)
        pprint('error: {}'.format(error))

    else:
        for item in general_response['outputs']:
            video_id = item['input']['data']['image']['url'][23:34]
            thumbnail_tags = set()
            for tag in item['data']['concepts']:
                if tag['value'] > .9:
                    tag_string = tag['name'].strip().lower()
                    if not Tag.query.filter(Tag.tag == tag_string).first():
                        add_tag_data(tag_string)
                    thumbnail_tags.add(tag_string)
            image_info[video_id] = {'tags': thumbnail_tags}


    # Obtain nsfw score
    try:
        nsfw_response = app.models.get('nsfw-v1.0').predict(initialized_ClImages)
    except ApiError as e:
        error = json.loads(e.response.content)
        pprint('error: {}'.format(error))

    else:
        for item in nsfw_response['outputs']: #nsfw_r['outputs'] is a list
            video_id = item['input']['data']['image']['url'][23:34]
            nsfw_score = round(item['data']['concepts'][-1]['value'] * 100)
            image_info[video_id]['nsfw_score'] = nsfw_score


    # Obtain color data
    # try:
    #     color_response = app.models.get('color').predict(initialized_ClImages)
    # except ApiError as e:
    #     error = json.loads(e.response.content)
    #     pprint('error: {}'.format(error[-100:]))

    # else:
    #     for item in color_response['outputs']:
    #         video_id = item['input']['data']['image']['url'][23:34]
    #         color_tags = {}
    #         for color in item['data']['colors']: # item['data']['colors'] is a list
    #             if color['value'] > .2:
    #                 color_hex = color['w3c']['hex'].rstrip().lower()
    #                 color_name = color['w3c']['name'].rstrip().lower()
    #                 if not Color.query.filter(Color.hex_code == color_hex).first():
    #                     add_color_data(color_hex, color_name)
    #             image_info[video_id]['colors'] = color_tags

    return image_info


def add_to_database(image_info):
    """Assume image_info is a dictionary with video_ids as keys.
    Populate the image_analyses, tags_images, and tags_colors tables if appropriate."""

    for video_id in image_info:
        info = image_info[video_id]
        print(info)
        add_nsfw_image_data(video_id, info)        
        image_analysis_id = ImageAnalysis.query.filter(ImageAnalysis.video_id == video_id).first().image_analysis_id

        add_tag_image_data(video_id, info, image_analysis_id)
        # add_color_image_data(video_id, info, image_analysis_id)


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()
