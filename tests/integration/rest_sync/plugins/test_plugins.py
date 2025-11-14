from pinecone import Pinecone


class TestAssistantPlugin:
    def test_assistant_plugin(self):
        pc = Pinecone()
        pc.assistant.list_assistants()
        assert True, "This should pass without errors"
