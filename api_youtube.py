#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

"""

import os, requests, json, datetime, doctest
from model import *
import dateutil.parser
from isodate import parse_duration
from googleapiclient.errors import HttpError

GOOGLE_KEY = os.environ.get('GOOGLE_KEY')
YOUTUBE_URL = 'https://www.googleapis.com/youtube/v3/'


def get_channel_id_from_video_id(video_id):
    """Access the YouTube API to gather a video's channel_id.
    
    >>> get_channel_id_from_video_id('KcyoKIjayGk')
    'UCZ9YdClgOz-5qEvkc1s3j6Q'
    """
    part = 'videos?part=snippet&fields=items%2Fsnippet%2FchannelId&id='
    url = YOUTUBE_URL + part + video_id + '&key=' + GOOGLE_KEY

    response = requests.get(url).json()
    channel_id = response['items'][0]['snippet']['channelId']

    return channel_id


def get_info_by_youtube_id(youtube_id):
    """Access the YouTube API to gather information about videos and channels.
    
    >>> get_info_by_youtube_id('UCZ9YdClgOz-5qEvkc1s3j6Q')['items'][0]['snippet']['publishedAt']
    '2012-07-23T00:42:37.000Z'

    >>> get_info_by_youtube_id('KcyoKIjayGk')['items'][0]['snippet']['channelId']
    'UCZ9YdClgOz-5qEvkc1s3j6Q'

    """
    part = '&part=snippet%2CcontentDetails%2Cstatistics'
    if youtube_id[:2] == 'UC': # YouTube channel IDs all begin with 'UC'
        url = YOUTUBE_URL + 'channels?id=' + youtube_id + part + '&key=' + GOOGLE_KEY
    else:
        url = YOUTUBE_URL + 'videos?id=' + youtube_id + part + '&key=' + GOOGLE_KEY
    
    try:
        response = requests.get(url).json()
    except HttpError as e:
        response = {'error': e}
    else: 
        response['timestamp'] = datetime.datetime.utcnow()
    finally:
        return response


def parse_channel_data(response, channel_in_db=False):
    """Assume data is a dictionary of the raw JSON returned by the YouTube API.
    Return a condensed dictionary with the necessary info."""
    try:
        channel_data = {}

        channel_data['timestamp'] = response['timestamp']
        items = response['items'][0]
        channel_data['channel_id'] = items['id']

        # Get information for the channel_stats table
        channel_data['total_views'] = items['statistics']['viewCount']
        
        if items['statistics']['hiddenSubscriberCount']:
            channel_data['total_subscribers'] = None
        else:
            channel_data['total_subscribers'] = items['statistics']['subscriberCount']
        
        channel_data['total_videos'] = items['statistics']['videoCount']
        channel_data['total_comments'] = items['statistics']['commentCount']

        if not channel_in_db: # This data does not change over time
            channel_data['channel_title'] = items['snippet']['title']
            channel_data['channel_description'] = items['snippet']['description']
            created_at = items['snippet']['publishedAt']
            # Convert string into a datetime object
            channel_data['created_at'] = dateutil.parser.parse(created_at)
            channel_data['country_code'] = None # not all channels have a country code
            try: 
                channel_data['country_code'] = items['snippet']['country']
            except KeyError:
                pass
            # tk Question: do I need a finally block in order to ensure that the rest 
            #  gets executed?
            finally:
                return channel_data
        # print('success: parse_channel_data for' + str(channel_data['channel_id']))
        return channel_data
    except Exception as e:
        print(e)


def add_channel_data(channel_data):
    """Assume channel_data is a dictionary whose keys represent fields in the channels table.
    Add data to the channels table."""

    channel = Channel(channel_id=channel_data['channel_id'],
                      channel_title=channel_data['channel_title'],
                      channel_description=channel_data['channel_description'],
                      created_at=channel_data['created_at'],
                      country_code=channel_data['country_code'])
    
    channel_data['channel_description']

    db.session.add(channel)
    db.session.commit()

    # print('success: add_channel_data for' + str(channel_data['channel_id']))    
    add_channel_stats_data(channel_data)


def add_channel_stats_data(channel_data):
    """Assume channel_data is a dicionary with fields in the channel_stats table as keys.
    Add the data to the channel_stats table."""

    channel_stat = ChannelStat(channel_id=channel_data['channel_id'],
                               retrieved_at=channel_data['timestamp'],
                               total_subscribers=channel_data['total_subscribers'],
                               total_views=channel_data['total_views'],
                               total_videos=channel_data['total_videos'],
                               total_comments=channel_data['total_comments'])
    
    db.session.add(channel_stat)
    db.session.commit()
    # print('success: add_channel_stats_data for' + str(channel_data['channel_id']))


def add_tag_data(tags):
    """Assume tags is a list of strings that represent tags.
    Add them to the tags table.

    >>> tags = [' art', '50 cent', 'Peru', '@$$', 'help\n']
    >>> list(map(lambda x: x.strip().lower(), tags))
    ['art', '50 cent', 'peru', '@$$', 'help']

    """

    # tk this might not be necessary because all tags are strings -- verify!
    try:
        tags = list(map(lambda x: x.strip().lower(), tags))
    except AttributeError: # tags that are numbers would raise an AttributeError
        tags = list(filter(lambda x: type(x) == str, tags))
    finally:
        for tag_string in tags:
            if not Tag.query.filter(Tag.tag == tag_string).first(): #None if tag isn't in db
                add_tag = Tag(tag=tag_string)
                db.session.add(add_tag)
        db.session.commit()

        return tags # lowercase and stripped of whitespace


def add_tag_video_data(tags, video_id):
    """Use tags and video_id to populate the tags_videos association table."""

    for tag_item in tags:
        # print(tag_item)
        tag_id = Tag.query.filter(Tag.tag == tag_item).first().tag_id
        if not TagVideo.query.filter(TagVideo.tag_id == tag_id,
                                     TagVideo.video_id == video_id).first():
            tag_object = Tag.query.filter(Tag.tag == tag_item).first()
            tag_video = TagVideo(video_id=video_id,
                                 tag_id=tag_id)
            db.session.add(tag_video)
    db.session.commit()
    # print('success: add_tag_video_data for' + str(video_id))


def add_video_stats_data(video_data):

    video_stat = VideoStat(video_id=video_data['video_id'],
                           retrieved_at=video_data['timestamp'],
                           views=video_data['views'],
                           likes=video_data['likes'],
                           dislikes=video_data['dislikes'],
                           comments=video_data['comments'])
    db.session.add(video_stat)
    db.session.commit()
    # print('success: add_video_stats_data for' + str(video_data['video_id']))


def update_video_details(video_data):

    video = Video.query.filter(Video.video_id == video_data['video_id']).first()

    video.channel_id = video_data['channel_id']
    video.video_title = video_data['video_title']
    video.video_description = video_data['video_description']
    video.published_at = video_data['published_at']
    video.category_id = video_data['category_id']
    video.duration = video_data['duration']
    video.thumbnail_url = video_data['thumbnail_url']

    db.session.commit() # no need to db.session.add because it already exists in the db

    add_video_stats_data(video_data) 
    # print('success: update_video_details for' + str(video_data['video_id']))


def parse_video_data(response, video_details_in_db=False):
    """Assume response is a dictionary of raw JSON returned by the YouTube API.
    Return a condensed dictionary with the necessary info."""

    video_data = {}

    video_data['timestamp'] = response['timestamp']
    items = response['items'][0]

    video_data['video_id'] = items['id']

    # Get information for the video_stats table
    video_data['views'] = items['statistics']['viewCount']
    video_data['likes'] = items['statistics']['likeCount']
    video_data['dislikes'] = items['statistics']['dislikeCount']
    video_data['comments'] = items['statistics']['commentCount']

    if not video_details_in_db:

        duration = items['contentDetails']['duration'] # duration is a timedelta object
        video_data['duration'] = parse_duration(duration)

        video_data['category_id'] = items['snippet']['categoryId']
        video_data['channel_id'] = items['snippet']['channelId']
        video_data['video_title'] = items['snippet']['title']
        video_data['video_description'] = items['snippet']['description']

        published_at = items['snippet']['publishedAt']
        # Convert into a datetime object
        video_data['published_at'] = dateutil.parser.parse(published_at)
        
        # Need to make sure all tags exist in the tags table first due to foreign keys/referential integrity.
        try:
            video_data['tags'] = items['snippet']['tags'] # type(tags) is list -- better to make this a set?
        except KeyError:
            pass
        else:
            add_tag_video_data(add_tag_data(video_data['tags']), video_data['video_id']) # Add data to the tags_videos table

        thumbnail_data = items['snippet']['thumbnails'] # dictionary whose keys are thumbnail versions
        if 'standard' in thumbnail_data:
            video_data['thumbnail_url'] = thumbnail_data['standard']['url']
        elif 'high' in thumbnail_data:
            video_data['thumbnail_url'] = thumbnail_data['high']['url']
        elif 'medium' in thumbnail_data:
            video_data['thumbnail_url'] = thumbnail_data['medium']['url']
        elif 'default' in thumbnail_data:
            video_data['thumbnail_url'] = thumbnail_data['default']['url']
        else:
            video_data['thumbnail_url'] = ''

    # print('success: parse_video_data for' + str(video_data['video_id']))
    return video_data


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    app.app_context().push()

