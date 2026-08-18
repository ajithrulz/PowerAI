from pathlib import Path

import pandas as pd
from entsoe import Client
import ssl
import os
from src.config import ENTSOE_API_KEY


class ENTSOEDataClient:

    def __init__(self):        
        ssl._create_default_https_context = ssl._create_unverified_context
        self.client = Client(api_key=ENTSOE_API_KEY)
        os.environ['REQUESTS_CA_BUNDLE'] = 'C:\\tls\\ION_CA_Base64.cer'
        os.environ['SSL_CERT_FILE'] = 'C:\\tls\\ION_CA_Base64.cer'
        
    @staticmethod
    def _validate_dates(start_date, end_date):

        start = pd.Timestamp(start_date)

        end = pd.Timestamp(end_date)

        if start.tzinfo is None:
            start = start.tz_localize("Europe/Brussels")

        if end.tzinfo is None:
            end = end.tz_localize("Europe/Brussels")

        if start >= end:
            raise ValueError("start_date must be before end_date")

        return start, end

    def get_day_ahead_prices(
        self,
        zone,
        start_date,
        end_date
    ):

        start, end = self._validate_dates(
            start_date,
            end_date
        )

        df = self.client.prices.day_ahead(
            start,
            end,
            country=zone
        )

        df = df.rename(
            columns={
                "value": "price_eur_mwh"
            }
        )

        df["zone"] = zone

        return df

    def get_actual_load(
        self,
        zone,
        start_date,
        end_date
    ):

        start, end = self._validate_dates(
            start_date,
            end_date
        )

        df = self.client.load.actual(
            start,
            end,
            country=zone
        )

        df = df.rename(
            columns={
                "value": "load_mw"
            }
        )

        df["zone"] = zone

        return df

    def get_load_forecast(
        self,
        zone,
        start_date,
        end_date
    ):

        start, end = self._validate_dates(
            start_date,
            end_date
        )

        df = self.client.load.forecast(
            start,
            end,
            country=zone
        )

        df = df.rename(
            columns={
                "value": "load_forecast_mw"
            }
        )

        df["zone"] = zone

        return df

    def get_generation(
        self,
        zone,
        start_date,
        end_date
    ):

        start, end = self._validate_dates(
            start_date,
            end_date
        )

        df = self.client.generation.actual(
            start,
            end,
            country=zone
        )

        df["zone"] = zone

        return df

    def get_generation_forecast(
        self,
        zone,
        start_date,
        end_date
    ):

        start, end = self._validate_dates(
            start_date,
            end_date
        )

        df = self.client.generation.forecast(
            start,
            end,
            country=zone
        )

        df["zone"] = zone

        return df

    def get_cross_border_flow(
        self,
        country_from,
        country_to,
        start_date,
        end_date
    ):

        start, end = self._validate_dates(
            start_date,
            end_date
        )

        df = self.client.transmission.crossborder_flows(
            start,
            end,
            country_from=country_from,
            country_to=country_to
        )

        df = df.rename(
            columns={
                "value": "flow_mw"
            }
        )

        df["country_from"] = country_from
        df["country_to"] = country_to

        return df