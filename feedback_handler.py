import sqlite3
import os

class FeedbackHandler:
    def __init__(self, db_path='feedback.db'):
        self.db_path = db_path
        self.conn = None
        self.create_table()

    def create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback
            (id INTEGER PRIMARY KEY, 
            user_input TEXT, 
            bot_response TEXT, 
            is_helpful BOOLEAN, 
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
            ''')
            conn.commit()

    def add_feedback(self, user_input, bot_response, is_helpful):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO feedback (user_input, bot_response, is_helpful)
            VALUES (?, ?, ?)
            ''', (user_input, bot_response, is_helpful))
            conn.commit()

    def get_all_feedback(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM feedback')
            return cursor.fetchall()

    def get_helpful_feedback(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_input, bot_response FROM feedback WHERE is_helpful = 1')
            return cursor.fetchall()