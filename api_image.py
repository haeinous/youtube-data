#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""

import os, json
from pprint import pprint
from clarifai.rest import ClarifaiApp, ApiError
from clarifai.rest import Image as ClImage
from model import connect_to_db, db, ImageAnalysis, Tag, Color, TagImage, ColorImage
from all_api_calls import file_to_video_id_list

CLARIFAI_KEY = os.environ.get('CLARIFAI_KEY')


####### new/single ########
def add_clarifai_data(thumbnail_url):

    image_info = parse_image_data(thumbnail_url)
    if image_info:
        add_non_duplicate_data(image_info)




####### old/bulk ########
# Create a list of initialized Clarifai images (class ClImage) and call the Clarifai API

def add_tag_data(tag):
    """Assume tag is a tag in string form.
    Add it to the database."""

    tag = tag.strip().lower()
    add_tag = Tag(tag=tag)
    db.session.add(add_tag)
    db.session.commit()
    return tag


def add_color_data(hex_code, color_name, color_name_in_db=False):
    """Assume hex_code is a 7-character string that's a hex code.
    Add it to the database if it doesn't already exist."""
    if color_name_in_db:
        color = Color.query.filter(Color.hex_code == hex_code).first()
        color.color_name = color_name
    else:
        add_color = Color(hex_code=hex_code.strip().lower(),
                          color_name=color_name.strip().lower())
        db.session.add(add_color)
    db.session.commit()
    return hex_code


def add_tag_image_data(image_info, image_analysis_id):
    """Assume image_info is a dictionary with video_ids as keys and
    image_analysis_id is the int primary key for the image_analyses table.
    Populate the tags_images association table."""

    for video_id in image_info:
        tags = image_info[video_id]['tags'] # tags is a dictionary
        for tag_to_add in tags:
            tag_id = Tag.query.filter(Tag.tag == tag_to_add).first().tag_id
            if not TagImage.query.filter(TagImage.tag_id == tag_id,
                                         TagImage.image_analysis_id == image_analysis_id).first():
                tag_image = TagImage(tag_id=tag_id, 
                                     image_analysis_id=image_analysis_id)
                db.session.add(tag_image)
    db.session.commit()


def add_color_image_data(image_info, image_analysis_id):
    """Assume image_info is a dictionary with video_ids as keys and
    image_analysis_id is the int primary key for the image_analyses table.
    Populate the colors_images association table."""

    for video_id in image_info:
        colors = image_info[video_id]['colors'] # colors is a dictionary
        for color_to_add in colors:
            hex_code = Color.query.filter(Color.hex_code == color_to_add).first().hex_code
            if not ColorImage.query.filter(ColorImage.hex_code == hex_code,
                                           ColorImage.image_analysis_id == image_analysis_id).first():        
                color_image = ColorImage(hex_code=hex_code,
                                         image_analysis_id=image_analysis_id)
                db.session.add(color_image)
    db.session.commit()          


def add_image_analysis_data(image_info):
    """Assume image_info is a dictionary with video_ids as keys.
    Populate the image_analyses table."""

    for video_id in image_info:
        nsfw_score = round(image_info[video_id]['nsfw_score'], 3)
        image_analysis = ImageAnalysis(video_id=video_id,
                                       nsfw_score=nsfw_score)
        db.session.add(image_analysis)
        print('video_id={}, nsfw_score={}'.format(video_id,nsfw_score))
    db.session.commit()


def parse_image_data(image_urls):
    """Assume image_urls is a list of image urls.
    Return a dictionary of tags and nsfw scores associated with each thumbnail image
    (video_ids are keys).
    """

    # Initialize images.
    initialized_ClImages = list(map(ClImage, image_urls))
    app = ClarifaiApp(api_key=CLARIFAI_KEY)
    image_info = {}

        # Obtain and parse image tag data from Clarifai API.

    try:
        general_response = app.models.get('general-v1.3').predict(initialized_ClImages)
    except ApiError as e:
        error = json.loads(e.response.content) #tk more sophisticated error handling by removing problematic inputs
        pprint(error)

    else:
        for item in general_response['outputs']:
            video_id = item['input']['data']['image']['url'][23:34]
            thumbnail_tags = {}
            for tag in item['data']['concepts']:
                if tag['value'] > .9:
                    tag_to_add = tag['name']
                    if not Tag.query.filter(Tag.tag == tag_to_add).first():
                        tag_to_add = add_tag_data(tag_to_add)
                    thumbnail_tags[tag_to_add] = round(tag['value'], 3)
            image_info[video_id] = {'tags':thumbnail_tags}
        # print('image_info after step 1: ' + str(image_info))

        # Obtain nsfw score for an image.
        nsfw_response = app.models.get('nsfw-v1.0').predict(initialized_ClImages)

        for item in nsfw_response['outputs']: #nsfw_r['outputs'] is a list
            video_id = item['input']['data']['image']['url'][23:34]
            nsfw_score = round(item['data']['concepts'][-1]['value'], 3)
            image_info[video_id]['nsfw_score'] = nsfw_score
        # print('image_info after step 2: ' + str(image_info))

        # Obtain color data for an image.
        color_response = app.models.get('color').predict(initialized_ClImages)
        print('********color_response********\n')
        pprint(color_response)

        for item in color_response['outputs']:
            video_id = item['input']['data']['image']['url'][23:34]
            color_tags = {}
            print('colors: ' + str(item['data']['colors']))
            for color in item['data']['colors']: # item['data']['colors'] is a list
                if color['value'] > .2:
                    print('adding ' + str(color['w3c']))
                    color_add = color['w3c']['hex'].strip().lower()
                    color_name_add = color['w3c']['name'].strip().lower()
                    color_basequery = Color.query.filter(Color.hex_code == color_add)
                    if not color_basequery.first():
                        add_color_data(color_add, color_name_add)
                    # tk remove the color_name thing for production
                    if not color_basequery.filter(Color.color_name.is_(None)).first():
                        add_color_data(color_add, color_name_add, color_name_in_db=True)
                    color_tags[color_add] = str(round(color['value'], 3))
                else:
                    print('not adding ' + str(color['w3c']))
                image_info[video_id]['colors'] = color_tags
        print('image_info after step 3: ' + str(image_info))
        return image_info


def add_non_duplicate_data(image_info):
    """Assume image_info is a dictionary with video_ids as keys.
    Populate the image_analyses, tags_images, and tags_colors tables if appropriate."""

    for video_id in image_info:
        if not ImageAnalysis.query.filter(ImageAnalysis.video_id == video_id).first(): # if the query returns none
            # print('video_id {} being added to the image_analyses table'.format(video_id))
            add_image_analysis_data(image_info)
        # else:
            # print('video_id {} already in image_analyses table'.format(video_id))
        
        image_analysis_id = ImageAnalysis.query.filter(ImageAnalysis.video_id == video_id).first().image_analysis_id
        add_tag_image_data(image_info, image_analysis_id)
        add_color_image_data(image_info, image_analysis_id)
        print('added: ' + video_id)

# def add_clarifai_data(video_ids):

#     thumbnail_urls = make_thumbnail_urls(video_ids)
#     image_info = parse_image_data(thumbnail_urls)
#     if image_info:
#         add_non_duplicate_data(image_info)


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()


