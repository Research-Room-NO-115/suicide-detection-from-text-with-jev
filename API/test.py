import requests

class Jev:
    def __init__(self, text):
        print('Jev is cnstructed!')
        self.url = "https://api.metisai.ir/typesafe/v1/systemone"
        self.headers = {
            "Authorization": "Bearer tpsg-gmA6u6whzQ8oGqbtk3dn1hyqzHf5C0m",
            "Content-Type": "application/json",
        }
        self.payload = {
            "state": text,
            "model": "jev-latest",
            "questions": {
                "department": {
                    "type": "score",
                    "instructions": "On a scale of 0 to 1, how likely is it that this text indicates a risk of suicidal intent?",
                    "criteria": [
                        "0 - No risk indicated",
                        "1 - High risk indicated"
                    ]
                },
            },
        }

        self.response = requests.post(self.url, headers=self.headers, json=self.payload)
        self.response.raise_for_status()

    def get_response(self):
        data = self.response.json()
        score = data['answers']['department']['score']

        return score
