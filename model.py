#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Hae-in Lim, haeinous@gmail.com

Models and database functions for Hae-in's Hackbright project."""


from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


#####################################################################
# Model definitions

class Video(db.Model):
    """A video on YouTube."""
    __tablename__ = 'videos'

    video_id = db.Column(db.String(11),
                         primary_key=True,
                         unique=True)
    ad_status_id = db.Column(db.Integer,
                             db.ForeignKey('ad_statuses.ad_status_id'))
    channel_id = db.Column(db.String(24), 
                           db.ForeignKey('channels.channel_id'))
    video_title = db.Column(db.String(100))
    video_description = db.Column(db.Text)
    published_at = db.Column(db.DateTime(timezone=False))
    category_id = db.Column(db.Integer,
                            db.ForeignKey('video_categories.video_category_id'))
    live_broadcast_id = db.Column(db.Integer,
                                  db.ForeignKey('live_broadcasts.live_broadcast_id'))
    duration = db.Column(db.Interval)
    # thumbnail_url = db.Column(db.String(48))
    # For user submissions through the /contribute.html -- this may become its own table
    # submitted_at = db.Column(db.DateTime(timezone=False))
    # updated_at = db.Column(db.DateTime(timezone=False))

    channel = db.relationship('Channel',
                              backref=db.backref('videos'))
    video_category = db.relationship('VideoCategory',
                                     backref=db.backref('videos'))

    def __repr__(self):
        return '<Video video_id={} channel_id={} ad_status_id={}>'.format(
                self.video_id,
                self.channel_id,
                self.ad_status_id)


class AdStatus(db.Model):
    """A video's ad status - four possibilities:
       (1) disabled by user
       (2) demonetized by youtube
       (3) monetized without adsense
       (4) fully monetized
    """
    __tablename__ = 'ad_statuses'

    ad_status_id = db.Column(db.Integer,
                             autoincrement = True,
                             primary_key=True)
    ad_status_name = db.Column(db.String(25))

    def __repr__(self):
        return '<AdStatus {} (id: {})>'.format(
                self.ad_status_name,
                self.ad_status_id)


class LiveBroadcast(db.Model):
    """A video can have one of three live broadcast statuses:
       (1) none,
       (2) upcoming, and
       (3) live """
    __tablename__ = 'live_broadcasts'

    live_broadcast_id = db.Column(db.Integer,
                                  primary_key=True,
                                  autoincrement=True)
    broadcast_status_name = db.Column(db.String(10))

    def __repr__(self):
        return '<LiveBroadcast {} (id: {})>'.format(
                self.live_broadcast_id,
                self.broadcast_status_name)


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
                         db.ForeignKey('videos.video_id'))
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
    channel_title = db.Column(db.String(100), 
                              nullable=False)
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


class ChannelPerson(db.Model):
    """An association table connecting the Channel and Person tables."""
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
    person_name = db.Column(db.String(100))

    channels = db.relationship('Channel',
                               secondary='channels_people',
                               backref=db.backref('people'))

    def __repr__(self):
        return '<Person {}, id={}>'.format(
                self.person_name, 
                self.person_id)


class ChannelStat(db.Model):

    __tablename__ = 'channel_stats'

    channel_stat_id = db.Column(db.Integer, 
                                autoincrement=True, 
                                primary_key=True)
    channel_id = db.Column(db.String(24), 
                           db.ForeignKey('channels.channel_id'))
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
    """Countries and ISO two-letter country codes."""
    __tablename__ = 'countries'

    country_code = db.Column(db.String(2), 
                             primary_key=True)
    country_name = db.Column(db.String(58))
    has_yt_space = db.Column(db.Boolean)

    def __repr__(self):
        return '<Country code={} name={}>'.format(self.country_code,
                                                  self.country_name)


class TextAnalysis(db.Model):
    __tablename__ = 'text_analyses'

    text_analysis_id = db.Column(db.Integer, 
                                 autoincrement=True, 
                                 primary_key=True)
    video_id = db.Column(db.String(11))
    textfield_id = db.Column(db.Integer, 
                             db.ForeignKey('textfields.textfield_id'))
    sentiment_score = db.Column(db.Float)
    sentiment_magnitude = db.Column(db.Float)
    # tk remove after drop/create db
    # sentiment_score_standard_deviation = db.Column(db.Float)
    # sentiment_max_score = db.Column(db.Float)
    # sentiment_min_score = db.Column(db.Float)
    language_code = db.Column(db.String(2),
                         db.ForeignKey('languages.language_code'))

    textfield = db.relationship('Textfield',
                                backref=db.backref('text_analyses'))
    language = db.relationship('Language',
                               backref=db.backref('languages'))

    def __repr__(self):
        return "<TextAnalysis for field {} of video {}\nscore: {}, magnitude: {}>".format(
                self.field_name,
                self.video_id,
                self.sentiment_score,
                self.sentiment_magnitude)


class Language(db.Model):
    __tablename__ = 'languages'

    language_code = db.Column(db.String(2),
                              primary_key=True)
    language_name = db.Column(db.String(10))

    def __repr__(self):
        return '<Language code={} name={}'.format(
                self.language_code,
                self.language_name)


class Textfield(db.Model):
    """
        Primarily exists to assert referential integrity in the
        TextAnalysis Table.
        video_title = 1
        video_description = 2
        tags = 3
    """
    __tablename__ = 'textfields'

    textfield_id = db.Column(db.Integer,
                             autoincrement=True,
                             primary_key=True)
    textfield_name = db.Column(db.String(17))

    def __repr__(self):
        return '<Textfield name={}, id={}>'.format(
                self.textfield_name,
                self.textfield_id)


class ImageAnalysis(db.Model):
    __tablename__ = 'image_analyses'

    image_analysis_id = db.Column(db.Integer,
                                  autoincrement=True,
                                  primary_key=True)
    video_id = db.Column(db.String(11), 
                         db.ForeignKey('videos.video_id'))
    nsfw_score = db.Column(db.Float)

    video = db.relationship('Video',
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
    tag = db.Column(db.String(100))

    videos = db.relationship('Video',
                             secondary='tags_videos',
                             backref=db.backref('tags'))
    image_analyses = db.relationship('ImageAnalysis',
                                     secondary='tags_images',
                                     backref=db.backref('tags'))

    def __repr__(self):
        return '<Tag tag={} id={}>'.format(
                self.tag,
                self.tag_id)


class TagVideo(db.Model):
    """An association table connecting the Tag and Video tables."""
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


class TagImage(db.Model):
    """An association table connecting the Tag and Image tables."""
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
    """An association table connecting the Color and ImageAnalysis tables."""
    __tablename__ = 'colors_images'

    color_image_id = db.Column(db.Integer,
                               autoincrement=True,
                               primary_key=True)
    color_hex_code = db.Column(db.String(7),
                     db.ForeignKey('colors.color_hex_code'))
    image_analysis_id = db.Column(db.Integer,
                        db.ForeignKey('image_analyses.image_analysis_id'))

    def __repr__(self):
        return '<ColorImage id={}>'.format(
                self.color_image_id)


class Color(db.Model):
    __tablename__ = 'colors'

    color_hex_code = db.Column(db.String(7),
                               primary_key=True)
    color_name = db.Column(db.String(25))

    def __repr__(self):
        return '<Color id={}, name={}>'.format(
                self.color_hex_code,
                self.color_name)





#####################################################################
# Helper functions

def connect_to_db(app):
    """Connect the database to the Flask app."""

    # Configure to use our PostgreSQL database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql:///youtube'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.app = app
    db.init_app(app)


if __name__ == '__main__':

    from server import app
    connect_to_db(app)
    db.create_all()
    print('Successfully connected to DB.')
