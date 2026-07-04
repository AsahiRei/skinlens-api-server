import unittest

from utils.chatbot import get_response


class ChatbotTests(unittest.TestCase):
    def test_returns_fallback_for_unknown_intent(self):
        response = get_response("unknown_intent")
        self.assertIn("Sorry", response)


if __name__ == "__main__":
    unittest.main()
