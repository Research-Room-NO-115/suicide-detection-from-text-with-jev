"""Compatibility wrapper around the study's budgeted Jev client.

Supply an authorized selected sample and shared client explicitly. Credentials
come from METIS_API_KEY; this module never contains or loads a source-code key.
"""
from jev_research.client import BudgetedJevClient


class Jev:
    def __init__(self, text: str, *, client: BudgetedJevClient, sample_id: str):
        if not isinstance(client, BudgetedJevClient):
            raise TypeError('Use a BudgetedJevClient with the shared study ledger.')
        self.score = client.predict(sample_id, text)

    def get_response(self) -> float:
        return self.score
