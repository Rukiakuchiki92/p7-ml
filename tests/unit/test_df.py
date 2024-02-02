import logging

import pandas as pd
import pytest

from main import df

# FN = ""


@pytest.fixture
def df():
    from main import df

    return df


class TestDf:
    """ """

    def test_load_df(self, df):
        """ """

        assert isinstance(df, pd.DataFrame)

    def test_id_client_list(self, df):
        """ """

        clients_id_list = df["ID_CLIENT"].tolist()
        assert isinstance(clients_id_list, list)

        assert len(clients_id_list)

        logging.warning(f"liste : {clients_id_list[:10]}")

        for _id in clients_id_list:
            assert isinstance(_id, int)
