import unittest
from app import app, db

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test variables."""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = app.config.get('DATABASE_URL') # Use existing dev DB for now
        # For a separate test DB:
        # self.app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://testuser:testpass@localhost/test_db'
        self.client = self.app.test_client()
        self._ctx = self.app.app_context()
        self._ctx.push()
        db.create_all() # Ensure tables are created (if using a clean test DB)

    def tearDown(self):
        """Executed after each test."""
        db.session.remove()
        # If using a persistent dev DB, be careful with db.drop_all()
        # For a dedicated test DB, this would be appropriate:
        # db.drop_all()
        self._ctx.pop()

if __name__ == '__main__':
    unittest.main()
