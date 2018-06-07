#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

Models and database functions for Hae-in's Hackbright project."""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index

db = SQLAlchemy()


#####################################################################
# Model definitions

class Video(db.Model):
    """A video on YouTube."""
    __tablename__ = 'videos'

    video_id = db.Column(db.String(11),
                         primary_key=True,
                         unique=True)
    is_monetized = db.Column(db.Boolean)
    channel_id = db.Column(db.String(24), 
                           db.ForeignKey('channels.channel_id'))
    video_title = db.Column(db.String(255))
    video_description = db.Column(db.Text)
    published_at = db.Column(db.DateTime(timezone=False))
    video_category_id = db.Column(db.Integer,
                                  db.ForeignKey('video_categories.video_category_id'))
    duration = db.Column(db.Interval)
    thumbnail_url = db.Column(db.String(255),
                              unique=True)
    video_status = db.Column(db.String(15))

    channel = db.relationship('Channel',
                              backref=db.backref('videos'))
    video_category = db.relationship('VideoCategory',
                                     backref=db.backref('videos'))

    def __repr__(self):
        return '<Video video_id={} channel={} is_monetized={}>'.format(
                self.video_id,
                self.channel_id,
                self.is_monetized)


class VideoCategory(db.Model):
    __tablename__ = 'video_categories'

    video_category_id = db.Column(db.Integer,
                                  primary_key=True)
    category_name = db.Column(db.String(24))

    def __repr__(self):
        return '<VideoCategory id={} name={}'.format(
                self.video_category_id, 
                self.category_name)


class VideoStat(db.Model):
    """Statistics about a particular YouTube video at a point in time."""
    __tablename__ = 'video_stats'

    video_stat_id = db.Column(db.Integer,
                              autoincrement=True, 
                              primary_key=True)
    video_id = db.Column(db.String(11), 
                         db.ForeignKey('videos.video_id'),
                         nullable=False)
    retrieved_at = db.Column(db.DateTime(timezone=False))
    views = db.Column(db.Integer)
    likes = db.Column(db.Integer)
    dislikes = db.Column(db.Integer)
    comments = db.Column(db.Integer)

    video = db.relationship('Video',
                            backref=db.backref('video_stats'))

    def __repr__(self):
        return '<VideoStat id={}, retrieved_at={}\nviews={} likes={} dislikes={} comments={}>'.format(
                self.video_id,
                self.retrieved_at,
                self.views,
                self.likes,
                self.dislikes,
                self.comments)


class Channel(db.Model):
    """YouTube channel."""
    __tablename__ = 'channels'

    channel_id = db.Column(db.String(24), 
                           primary_key=True)
    channel_title = db.Column(db.String(255))
    channel_description = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=False))
    country_code = db.Column(db.String(2),
                             db.ForeignKey('countries.country_code'))

    country = db.relationship('Country',
                              backref=db.backref('channels'))

    def __repr__(self):
        return "<Channel id={} title={}>".format(
                self.channel_id,
                self.channel_title)


class ChannelStat(db.Model):

    __tablename__ = 'channel_stats'

    channel_stat_id = db.Column(db.Integer, 
                                autoincrement=True, 
                                primary_key=True)
    channel_id = db.Column(db.String(24), 
                           db.ForeignKey('channels.channel_id'),
                           nullable=False)
    retrieved_at = db.Column(db.DateTime(timezone=False))
    total_subscribers = db.Column(db.Integer)
    total_views = db.Column(db.BigInteger)
    total_videos = db.Column(db.Integer)
    total_comments = db.Column(db.Integer)

    channel = db.relationship('Channel',
                              backref=db.backref('channel_stats'))

    def __repr__(self):
        return '<ChannelStat channel_id={}\ntotal_subs={}, total_views={}\nretrieved_at={}>'.format(
                self.channel_id,
                self.total_subscribers,
                self.total_views,
                self.retrieved_at)


class Country(db.Model):
    """Countries and two-letter ISO country codes."""
    __tablename__ = 'countries'

    country_code = db.Column(db.String(2), 
                             primary_key=True)
    country_name = db.Column(db.String(255))
    has_yt_space = db.Column(db.Boolean)

    def __repr__(self):
        return '<Country code={} name={}>'.format(self.country_code,
                                                  self.country_name)


class TextAnalysis(db.Model):
    __tablename__ = 'text_analyses'
    __table_args__ = (Index('ix_unique_text_analysis',
                            'textfield', 'video_id',
                            unique=True),) # Ensures only one analysis per video-textfield pair

    text_analysis_id = db.Column(db.Integer, 
                                 autoincrement=True, 
                                 primary_key=True)
    video_id = db.Column(db.String(11),
                         db.ForeignKey('videos.video_id'))
    channel_id = db.Column(db.String(24),
                           db.ForeignKey('channels.channel_id'),
                           unique=True)
    textfield = db.Column(db.String(255))
    sentiment_score = db.Column(db.Float)
    sentiment_magnitude = db.Column(db.Float)
    sentiment_score_standard_deviation = db.Column(db.Float)
    sentiment_max_score = db.Column(db.Float)
    sentiment_min_score = db.Column(db.Float)
    language_code = db.Column(db.String(4))

    video = db.relationship('Video',
                            backref=db.backref('text_analyses'))
    channel = db.relationship('Channel',
                              backref=db.backref('text_analyses'))


    def __repr__(self):
        return '<TextAnalysis for field {} of {}{}\nscore: {}, magnitude: {}>'.format(
                self.textfield,
                self.video_id,
                self.channel_id,
                self.sentiment_score,
                self.sentiment_magnitude)


class ImageAnalysis(db.Model):
    __tablename__ = 'image_analyses'

    image_analysis_id = db.Column(db.Integer,
                                  autoincrement=True,
                                  primary_key=True)
    video_id = db.Column(db.String(11), 
                         db.ForeignKey('videos.video_id'),
                         unique=True)
    nsfw_score = db.Column(db.Integer) # Score is from 0 to 100
    video = db.relationship('Video',
                            backref=db.backref('image_analyses'))

    colors = db.relationship('Color',
                             secondary='colors_images',
                             backref=db.backref('image_analyses'))

    def __repr__(self):
        return '<ImageAnalysis id={} video={} nsfw_score={}>'.format(
                self.image_analysis_id,
                self.video_id,
                self.nsfw_score)


class Tag(db.Model):
    """A table containing all the tags from video tags and image tags."""
    __tablename__ = 'tags'

    tag_id = db.Column(db.Integer,
                       autoincrement=True,
                       primary_key=True)
    tag = db.Column(db.String(255),
                    unique=True)

    videos = db.relationship('Video',
                             secondary='tags_videos',
                             backref=db.backref('tags'))
    image_analyses = db.relationship('ImageAnalysis',
                                     secondary='tags_images',
                                     backref=db.backref('tags'))
    channels = db.relationship('Channel',
                               secondary='tags_channels',
                               backref=db.backref('channels'))

    def __repr__(self):
        return '<Tag tag={} id={}>'.format(
                self.tag,
                self.tag_id)


class TagVideo(db.Model):
    """An association table connecting the tags and videos tables."""
    __tablename__ = 'tags_videos'

    tag_video_id = db.Column(db.Integer,
                             autoincrement=True,
                             primary_key=True)
    tag_id = db.Column(db.Integer,
                       db.ForeignKey('tags.tag_id'))
    video_id = db.Column(db.String(11),
                         db.ForeignKey('videos.video_id'))

    def __repr__(self):
        return '<TagVideo id={}>'.format(
                self.tag_video_id)


class TagChannel(db.Model):
    """An association table connecting the tags and channels tables."""
    __tablename__ = 'tags_channels'

    tag_channel_id = db.Column(db.Integer,
                               autoincrement=True,
                               primary_key=True)
    tag_id = db.Column(db.Integer,
                       db.ForeignKey('tags.tag_id'))
    channel_id = db.Column(db.String(11),
                           db.ForeignKey('channels.channel_id'))

    def __repr__(self):
        return '<TagChannel id={}>'.format(self.tag_channel_id)


class TagImage(db.Model):
    """An association table connecting the tags and images tables."""
    __tablename__ = 'tags_images'

    tag_image_id = db.Column(db.Integer,
                             autoincrement=True,
                             primary_key=True)
    image_analysis_id = db.Column(db.Integer,
                        db.ForeignKey('image_analyses.image_analysis_id'))
    tag_id = db.Column(db.Integer,
                       db.ForeignKey('tags.tag_id'))

    def __repr__(self):
        return '<TagImage id={}>'.format(
                self.tag_image_id)


class ColorImage(db.Model):
    """An association table connecting the color and image_analyses tables."""
    __tablename__ = 'colors_images'

    color_image_id = db.Column(db.Integer,
                               autoincrement=True,
                               primary_key=True)
    hex_code = db.Column(db.String(7),
                         db.ForeignKey('colors.hex_code'))
    image_analysis_id = db.Column(db.Integer,
                        db.ForeignKey('image_analyses.image_analysis_id'))

    def __repr__(self):
        return '<ColorImage id={}>'.format(
                self.color_image_id)


class Color(db.Model):
    __tablename__ = 'colors'

    hex_code = db.Column(db.String(7),
                         primary_key=True)
    color_name = db.Column(db.String(255))

    def __repr__(self):
        return '<Color id={}, name={}>'.format(
                self.hex_code,
                self.color_name)


class Document(db.Model):
    """Document IDs and details for the inverted index."""
    __tablename__ = 'documents'

    document_id = db.Column(db.Integer,
                            autoincrement=True,
                            primary_key=True)
    document_type = db.Column(db.String, 
                              nullable=False) # channel, video, tag, or category
    document_primary_key = db.Column(db.String(24))
    document_subtype = db.Column(db.String) # title, description, video tags, image tags

    def __repr__(self):
        return '<Document: id={}, type={}>'.format(self.document_id,
                                                   self.document_type)

class TagChart(db.Model):
    """A table with info about which tags users are adding to the chart."""
    __tablename__ = 'tagcharts'

    tag_chart_id = db.Column(db.Integer,
                             autoincrement=True,
                             primary_key=True)
    tag_id = db.Column(db.Integer,
             db.ForeignKey('tags.tag_id'))
    added_on = db.Column(db.DateTime(timezone=False))

    tag = db.relationship('Tag',
                          backref=db.backref('tagcharts'))


class Addition(db.Model):
    """A table with info on data updates/additions by users."""
    __tablename__ = 'additions'

    addition_id = db.Column(db.Integer,
                            autoincrement=True,
                            primary_key=True)
    video_id = db.Column(db.String(11),
               db.ForeignKey('videos.video_id'),
               nullable=False)
    added_on = db.Column(db.DateTime(timezone=False))
    is_update = db.Column(db.Boolean)

    video = db.relationship('Video',
                            backref=db.backref('additions'))

    def __repr__(self):
        return '<Addition video_id={}, date={}>'.format(
                self.addition_id,
                self.added_on)


class Search(db.Model):
    """Searches on website."""
    __tablename__ = 'searches'

    search_id = db.Column(db.Integer,
                          autoincrement=True,
                          primary_key=True)
    searched_on = db.Column(db.DateTime(timezone=False))
    search_text = db.Column(db.String(255))


class Prediction(db.Model):
    """Predictions about monetization status."""
    __tablename__ = 'predictions'

    prediction_id = db.Column(db.Integer,
                              autoincrement=True,
                              primary_key=True)
    channel_id = db.Column(db.String(24),
                 db.ForeignKey('channels.channel_id'))
    predicted_monetization_status = db.Column(db.Boolean)

    field1 = db.Column(db.Text)
    field2 = db.Column(db.Text)
    field3 = db.Column(db.Text)
    field4 = db.Column(db.Integer)
    field5 = db.Column(db.Integer)

    channel = db.relationship('Channel',
                              backref=db.backref('predictions'))


class ChannelPerson(db.Model):
    """An association table connecting the channels and people tables."""
    __tablename__ = 'channels_people'

    channel_person_id = db.Column(db.Integer,
                                  autoincrement=True,
                                  primary_key=True)
    channel_id = db.Column(db.String(24),
                           db.ForeignKey('channels.channel_id'),
                           nullable=False)
    person_id = db.Column(db.Integer,
                          db.ForeignKey('people.person_id'),
                          nullable=False)


    def __repr__(self):
        return '<ChannelPerson id={}>'.format(self.channel_person_id)


class Person(db.Model):

    __tablename__ = 'people'

    person_id = db.Column(db.Integer,
                          autoincrement=True,
                          primary_key=True)
    person_name = db.Column(db.String(255),
                            unique=True)

    field1 = db.Column(db.String(255))
    field2 = db.Column(db.String(255))
    field3 = db.Column(db.String(255))
    field4 = db.Column(db.Integer)
    field5 = db.Column(db.Text)
    field6 = db.Column(db.Text)
    field7 = db.Column(db.String(255))
    field8 = db.Column(db.String(255))

    channel = db.relationship('Channel',
                              secondary='channels_people',
                              backref=db.backref('people'))

    def __repr__(self):
        return '<Person {}, id={}>'.format(
                self.person_name, 
                self.person_id)



#####################################################################
# Helper functions

def connect_to_db(app, uri='postgresql:///youtube'):
    """Connect database to the Flask app."""

    # Configure to use PostgreSQL.
    app.config['SQLALCHEMY_DATABASE_URI'] = uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.app = app
    db.init_app(app)


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    db.create_all()
    print('Successfully connected to DB.')