"""Deterministic provider for tests only; no configuration or network access."""


class FakeProvider:
    def __init__(self, output='{"status":"insufficient_context","paragraphs":[]}', error=None):
        self.output = output
        self.error = error
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.output
