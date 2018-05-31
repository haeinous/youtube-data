from unittest import TestCase
from model import connect_to_db, db, example_data
from server import app
from flask import session

# https://github.com/mjhea0/flaskr-tdd#jquery

class FlaskTestsBasic(TestCase):
    """Flask tests."""

    def setUp(self):
        """Stuff to do before every test."""

        self.client = app.test_client()
        app.config['TESTING'] = True

    def test_index(self):
        """Test homepage."""

        result = self.client.get('/')
        self.assertIn('read more »', result.data)


class FlaskTestsDatabase(TestCase):
    """Flask tests that use the database."""

    def setUp(self):
        """Stuff to do before every test."""

        self.client = app.test_client()
        app.config['TESTING'] = True

        connect_to_db(app, 'postgresql:///youtube')
        db.create_all()
        example_data()

    def tearDown(self):
        """Do at end of every test."""

        db.session.close()
        db.drop_all()

    def test_about(self):
        """Test about page."""

        result = self.client.get('/about')
        self.assertIn('Sapna Maheshwari', result.data)

    # tk add unique text from each page

    def test_add_data(self):
        """Test the add-data page."""

        # Test for Video ID character length validation.
        result1 = self.client.post('/add-data',
                                   data={'video_id': '123'})
        self.assertIn('11 characters long', result1.data)

        # Test to make sure video IDs only contain legal characters.
        result2 = self.client.post('/add-data',
                                   data={'video_id': '$RPRDK7jEns'})
        self.assertIn('alphanumeric', result2.data)


class FlaskTestsAjax(TestCase):

    def setUp(self):
        """Stuff to do before every test."""

        self.client = app.test_client()
        app.config['TESTING'] = True

        connect_to_db(app, 'postgresql:///youtube')
        db.create_all()
        example_data()

    def tearDown(self):
        """Do at end of every test."""

        db.session.close()
        db.drop_all()

    def test_autocomplete_min_length(self):
        """Test that jQuery isn't sending any searches that are
        shorter than three words."""

        result = self.



if __name__ == '__main__':
    import unittest

    unittest.main()
