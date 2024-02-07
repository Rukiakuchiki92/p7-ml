import logging

import pytest
from fastapi.testclient import TestClient

from main import app

# LIST_IDS = [
#     196888,
#     435135,
#     396314,
#     341153,
#     145376,
#     339498,
#     157136,
#     101392,
# ]


LIST_IDS = [
    196888,
    101392,
    435135,
    396314,
    341153,
    145376,
    339498,
    157136,
    420137,
    360918,
]


@pytest.fixture
def client():
    return TestClient(app)


class TestApi:
    """Test the api"""

    # utiliser la convention de nommage de test (gherkin) 
    def test_root(self, client):
        """Test the / endpoint"""

        # make api call
        response = client.get("/")

        # status code
        assert response.status_code == 200

        # json
        msg = {"Hello": "World"}
        assert response.json() == msg

    # utiliser la convention de nommage de test (gherkin) 
    def test_get_list_ids(self, client):
        """Test the get_list_ids endpoint"""

        # make api call
        response = client.get("/get_list_ids")

        # status code
        assert response.status_code == 200

        # json
        json = response.json()
        assert isinstance(json, dict)

        # payload
        li = json["list_ids"]
        assert isinstance(li, list)
        assert li

    # utiliser la convention de nommage de test (gherkin) 
    def test_get_population_summary(self, client):
        """Test the get_population_summary endpoint"""

        # make api call
        response = client.get("/get_population_summary")

        # status code
        assert response.status_code == 200

        # json
        json = response.json()
        assert isinstance(json, dict)

        # TODO : enable this one

        # # payload
        # li = json["list_ids"]
        # assert isinstance(li, list)
        # assert li

    # utiliser la convention de nommage de test (gherkin) 
    @pytest.mark.parametrize("client_id", LIST_IDS)
    def test_get_client_info(self, client, client_id):
        """Test the get_client_info endpoint"""

        # make api call
        response = client.get(f"/get_client_info/{client_id}")

        # status code
        assert response.status_code == 200

        # json
        json = response.json()
        assert isinstance(json, dict)

        # TODO : enable this one

        logging.warning(f"{json}")

        # # payload
        # li = json["list_ids"]
        # assert isinstance(li, list)
        # assert li
