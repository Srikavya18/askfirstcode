import os
import sys
import unittest
from fastapi.testclient import TestClient
# Add backend directory to Python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.append(backend_path)
# Set database url to a test database in the root folder
os.environ["DATABASE_URL"] = "sqlite:///../../test_chat_history.db"
# Import backend components
import main
from main import app
from database import init_db, SessionLocal, Thread, Message
class TestUniversalMemoryChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the database
        init_db()
        cls.client = TestClient(app)
    def setUp(self):
        # Clear database before each test
        db = SessionLocal()
        db.query(Message).delete()
        db.query(Thread).delete()
        db.commit()
        db.close()
    def test_universal_memory_flow(self):
        # 1. Create Thread 1 (Personal Info)
        response = self.client.post("/threads", json={"id": "thread-1", "title": "Personal Info"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "thread-1")
        # Send info to Thread 1
        response = self.client.post("/threads/thread-1/messages", json={"content": "My name is John"})
        self.assertEqual(response.status_code, 200)
        
        # 2. Create Thread 2 (Travel Info)
        response = self.client.post("/threads", json={"id": "thread-2", "title": "Travel Info"})
        self.assertEqual(response.status_code, 200)
        # Send info to Thread 2
        response = self.client.post("/threads/thread-2/messages", json={"content": "I am planning a trip to Paris"})
        self.assertEqual(response.status_code, 200)
        # 3. Create Thread 3 (Recall Query)
        response = self.client.post("/threads", json={"id": "thread-3", "title": "Recall Query"})
        self.assertEqual(response.status_code, 200)
        # Query Thread 3 about information in Thread 1 & Thread 2
        response = self.client.post("/threads/thread-3/messages", json={"content": "What is my name and where am I going?"})
        self.assertEqual(response.status_code, 200)
        
        assistant_reply = response.json()["assistant_message"]["content"]
        print(f"\n--- Test Recall Response ---\n{assistant_reply}\n----------------------------")
        
        # Verify the smart mock provider successfully extracted and integrated cross-thread memory
        self.assertTrue("John" in assistant_reply, "Mock response should recall name 'John' from Thread 1")
        self.assertTrue("Paris" in assistant_reply, "Mock response should recall location 'Paris' from Thread 2")
if __name__ == "__main__":
    unittest.main()
