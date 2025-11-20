"""
Kümeleme ve regresyon deneyleri için veri yükleyici yardımcı fonksiyonlar.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HOTEL_COLUMNS = [
    "id",
    "address",
    "city",
    "company_name",
    "country",
    "email_main",
    "iso_country_code",
    "last_update",
    "phone",
    "rooms",
    "source",
    "stars",
    "unsubscribed",
    "updated_by",
    "website_url",
    "data_class",
    "chains_brands",
    "email_2",
    "email_3",
    "geo_latitude",
    "geo_longitude",
    "hotel_information_url",
    "province",
    "region",
    "segment",
    "source_id",
    "town",
    "website_url_2",
    "zip_code",
    "town_city",
    "extra",
]

LEAD_SCORE_COLUMNS = [
    "id",
    "data_class",
    "average_score",
    "created_at",
    "hotel_email",
    "hotel_id",
    "hotel_name",
    "last_campaign_date",
    "max_score",
    "min_score",
    "total_campaigns",
    "total_score_sum",
    "updated_at",
]


def load_hotel_data():
    """
    Hotel_data verisini CSV’den okuyup DataFrame döndürür.
    """
    df = pd.read_csv(
        DATA_DIR / "hotel_data.csv",
        names=HOTEL_COLUMNS,
        header=None,
        engine="python",
        on_bad_lines="skip",
    )
    return df


def load_hotel_lead_scores():
    """
    hotel_lead_score verisini CSV’den okuyup DataFrame döndürür.
    """
    df = pd.read_csv(
        DATA_DIR / "hotel_lead_score.csv",
        names=LEAD_SCORE_COLUMNS,
        header=None,
        engine="python",
        on_bad_lines="skip",
    )
    return df
