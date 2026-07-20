import unittest

from studio.providers import OpenAICompatibleProvider


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self.content = content

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}

    def raise_for_status(self):
        return None


class FakeClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def post(self, url, json):
        self.calls.append((url, json))
        return FakeResponse(self.contents.pop(0))


class ProviderRetryTests(unittest.TestCase):
    def test_retries_malformed_json_without_reusing_bad_text(self):
        provider = OpenAICompatibleProvider("https://example.com/v1", "key")
        fake = FakeClient(['{"zh":"缺少结束"', '{"zh":"正确"}'])
        provider.client = fake
        result = provider.chat_json("model", "system", {"target": {"id": 1}})
        self.assertEqual(result["zh"], "正确")
        self.assertEqual(len(fake.calls), 2)
        retry_messages = fake.calls[1][1]["messages"]
        self.assertIn("invalid JSON", retry_messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
