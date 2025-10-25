#!/usr/bin/env python3
"""
Meta Ads Insights data fetcher for ad performance forecasting.

Builds an hourly campaign-level dataset from Meta Ads Insights API and flattens actions.

Outputs columns:
- campaign_name, hourly_window, spend, impressions, reach, clicks, unique_clicks, ctr, cpc, cpm, cpp
- web_in_store_purchase, add_to_cart, landing_page_view, video_view, post, post_reaction,
  initiate_checkout, post_interaction_gross, view_content, post_engagement, page_engagement

Usage:
  Set env vars or edit defaults:
    AD_ACCOUNT_ID="act_XXXXXXXXXXXXXXX"
    ACCESS_TOKEN="EAAXXXXX..."
    SINCE="2025-10-01"
    UNTIL="2025-10-01"
  Then run:
    python -m src.fetch.fetch_meta_insights
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Import from parent config
from config import config

# Graph API configuration
GRAPH_VERSION = "v19.0"
BASE_URL = f"https://graph.facebook.com/{GRAPH_VERSION}"

# Actions mapping: aggregate multiple Meta action_type variants into one column
ACTION_AGG_MAP = {
    # purchases happening in-store via web/app (rare but present in your payloads)
    "web_in_store_purchase": {
        "web_in_store_purchase",
        "web_app_in_store_purchase",
    },
    "add_to_cart": {
        "add_to_cart",
        "omni_add_to_cart",
        "onsite_web_add_to_cart",
        "onsite_web_app_add_to_cart",
        "offsite_conversion.fb_pixel_add_to_cart",
    },
    "landing_page_view": {
        "landing_page_view",
        "omni_landing_page_view",
    },
    "video_view": {
        "video_view",
    },
    "post": {
        "post",
    },
    "post_reaction": {
        "post_reaction",
    },
    "initiate_checkout": {
        "initiate_checkout",
        "omni_initiated_checkout",
        "onsite_web_initiate_checkout",
        "offsite_conversion.fb_pixel_initiate_checkout",
    },
    "post_interaction_gross": {
        "post_interaction_gross",
    },
    "view_content": {
        "view_content",
        "omni_view_content",
        "onsite_web_view_content",
        "onsite_web_app_view_content",
        "offsite_conversion.fb_pixel_view_content",
    },
    "post_engagement": {
        "post_engagement",
    },
    "page_engagement": {
        "page_engagement",
    },
}

# Final dataset column order
DATASET_COLUMNS = [
    "campaign_name",
    "date",
    "hourly_window",
    "spend",
    "impressions",
    "clicks",
    "ctr",
    "cpc",
    "cpm",
    "cpp",
    # flattened actions
    "web_in_store_purchase",
    "add_to_cart",
    "landing_page_view",
    "video_view",
    "post",
    "post_reaction",
    "initiate_checkout",
    "post_interaction_gross",
    "view_content",
    "post_engagement",
    "page_engagement",
]


def _to_num(x):
    """Safely cast Meta string numerics to float; return 0.0 for None/''."""
    if x is None or x == "":
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


class MetaInsightsFetcher:
    """Meta Ads Insights API fetcher with rate limiting and error handling."""

    def __init__(self):
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def fetch_insights_hourly(
        self,
        ad_account_id: str,
        access_token: str,
        since: str,
        until: str,
        level: str = "campaign",
        limit: int = 5000,
        sleep_sec: float = 0.2,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
    ):
        """
        Fetch hourly insights with hourly advertiser-time breakdown.
        Returns a list of raw rows (dicts) from the API.
        Implements pagination and rate limiting with exponential backoff.
        """
        fields = ",".join([
            "account_id",
            "account_name",
            "campaign_id",
            "campaign_name",
            "objective",
            "buying_type",
            "date_start",
            "date_stop",
            "spend",
            "impressions",
            "clicks",
            "ctr",
            "cpc",
            "cpm",
            "cpp",
            "cost_per_inline_link_click",
            "inline_link_clicks",
            "outbound_clicks",
            "cost_per_outbound_click",
            "unique_inline_link_clicks",
            "cost_per_unique_inline_link_click",
            "actions",
            "action_values",
            "unique_actions",
            "cost_per_action_type",
            "conversions",
            "conversion_values",
            "cost_per_conversion",
            "video_avg_time_watched_actions",
            "video_p25_watched_actions",
            "video_p50_watched_actions",
            "video_p75_watched_actions",
            "video_p95_watched_actions",
            "video_p100_watched_actions",
            "estimated_ad_recallers",
            "estimated_ad_recall_rate",
            "quality_ranking",
            "engagement_rate_ranking",
            "conversion_rate_ranking",
            "ad_delivery",
            "adset_id",
            "adset_name",
        ])

        base_params = {
            "level": level,
            "fields": fields,
            "time_range": json.dumps({"since": since, "until": until}),
            "time_increment": 1,
            "breakdowns": "hourly_stats_aggregated_by_advertiser_time_zone",
            "limit": limit,
            "access_token": access_token,
        }

        url = f"{BASE_URL}/{ad_account_id}/insights"
        rows = []
        page_count = 0
        total_rows = 0

        def make_request(url: str, params: Dict, retry_count: int = 0) -> requests.Response:
            """Make API request with retry logic and exponential backoff."""
            try:
                self.logger.info(f"Making API request (attempt {retry_count + 1})...")
                resp = requests.get(url, params=params, timeout=60)

                # Handle rate limiting (HTTP 429)
                if resp.status_code == 429:
                    if retry_count < max_retries:
                        wait_time = rate_limit_delay * (2 ** retry_count)  # Exponential backoff
                        self.logger.warning(f"Rate limited. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        return make_request(url, params, retry_count + 1)
                    else:
                        raise RuntimeError(f"Rate limit exceeded after {max_retries} retries")

                # Handle other HTTP errors
                if resp.status_code != 200:
                    if retry_count < max_retries:
                        wait_time = rate_limit_delay * (2 ** retry_count)
                        self.logger.warning(f"API error {resp.status_code}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        return make_request(url, params, retry_count + 1)
                    else:
                        raise RuntimeError(f"Insights API error {resp.status_code}: {resp.text}")

                return resp

            except requests.exceptions.RequestException as e:
                if retry_count < max_retries:
                    wait_time = rate_limit_delay * (2 ** retry_count)
                    self.logger.warning(f"Request error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    return make_request(url, params, retry_count + 1)
                else:
                    raise RuntimeError(f"Request failed after {max_retries} retries: {e}")

        while True:
            page_count += 1
            self.logger.info(f"Fetching page {page_count}...")

            resp = make_request(url, base_params if page_count == 1 else {})
            payload = resp.json()

            page_data = payload.get("data", [])
            rows.extend(page_data)
            total_rows += len(page_data)

            self.logger.info(f"Fetched {len(page_data)} rows (total: {total_rows})")

            # Additional check: if we got no data but there's still a next URL, we might be done
            if len(page_data) == 0 and next_url:
                self.logger.warning("Received empty data page but next URL exists. This might indicate the end of results.")
                break

            # Enhanced pagination handling with detailed logging
            paging = payload.get("paging", {})
            cursors = paging.get("cursors", {})
            next_url = paging.get("next")

            self.logger.info(f"Page {page_count} pagination info: has_next={bool(next_url)}")
            self.logger.info(f"Cursors: before={cursors.get('before', 'N/A')[:10]}..., after={cursors.get('after', 'N/A')[:10]}...")

            if next_url:
                # Parse the next URL to show pagination parameters for debugging
                try:
                    parsed_url = urlparse(next_url)
                    query_params = parse_qs(parsed_url.query)
                    pagination_params = {k: v[0] for k, v in query_params.items() if k in ['after', 'before', 'limit']}
                    self.logger.info(f"Next URL pagination params: {pagination_params}")
                except Exception as e:
                    self.logger.warning(f"Could not parse next URL params: {e}")

            if not next_url:
                self.logger.info(f"Completed fetching. Total pages: {page_count}, Total rows: {total_rows}")
                break

            # Validate next URL before using it
            if not next_url.startswith("https://graph.facebook.com/"):
                self.logger.warning(f"Invalid next URL format: {next_url}")
                break

            # Additional safety check: if we've fetched too many pages, break to prevent infinite loops
            if page_count > 1000:  # Reasonable upper limit
                self.logger.error(f"Too many pages fetched ({page_count}). Breaking to prevent infinite loop.")
                break

            url = next_url
            self.logger.info("Moving to next page...")

            # Small delay between pages to respect rate limits
            time.sleep(sleep_sec)

        return rows

    def flatten_row(self, row: dict) -> dict:
        """
        Produces one normalized record per API row, with:
        - campaign_name, date, hourly_window
        - numeric metrics
        - flattened action counts matching ACTION_AGG_MAP
        """
        # Extract date directly from date_start field (format: "YYYY-MM-DD")
        date_str = row.get("date_start", "")
        hourly_window = row.get("hourly_stats_aggregated_by_advertiser_time_zone", "")

        # Validate and ensure it's a proper date format (YYYY-MM-DD)
        if date_str and (not date_str.startswith("202") or len(date_str) != 10):
            # Fallback to extracting from hourly_window if date_start is invalid
            if hourly_window and 'T' in hourly_window:
                date_str = hourly_window.split('T')[0]
            else:
                date_str = ""

        out = {
            "campaign_name": row.get("campaign_name", ""),
            "date": date_str,
            "hourly_window": hourly_window,
            "spend": _to_num(row.get("spend")),
            "impressions": int(_to_num(row.get("impressions"))),
            "clicks": int(_to_num(row.get("clicks"))),
            "ctr": _to_num(row.get("ctr")),
            "cpc": _to_num(row.get("cpc")),
            "cpm": _to_num(row.get("cpm")),
            "cpp": _to_num(row.get("cpp")),
            # Initialize all action columns to 0
            "web_in_store_purchase": 0.0,
            "add_to_cart": 0.0,
            "landing_page_view": 0.0,
            "video_view": 0.0,
            "post": 0.0,
            "post_reaction": 0.0,
            "initiate_checkout": 0.0,
            "post_interaction_gross": 0.0,
            "view_content": 0.0,
            "post_engagement": 0.0,
            "page_engagement": 0.0,
        }

        # flatten actions[]
        for item in row.get("actions", []) or []:
            a_type = item.get("action_type")
            val = _to_num(item.get("value"))
            if not a_type:
                continue
            # add to any aggregate bucket that includes this action_type
            for col, variants in ACTION_AGG_MAP.items():
                if a_type in variants:
                    out[col] += val

        # cast action columns to int where appropriate (counts)
        for k in ACTION_AGG_MAP.keys():
            out[k] = int(out[k])

        return out

    def build_dataset(self, raw_rows):
        """Build pandas DataFrame from raw API rows."""
        records = [self.flatten_row(r) for r in raw_rows]
        df = pd.DataFrame.from_records(records, columns=DATASET_COLUMNS)

        # Ensure numeric dtypes (helps downstream ML)
        numeric_cols = [
            "spend", "impressions", "clicks",
            "ctr", "cpc", "cpm", "cpp",
            "web_in_store_purchase", "add_to_cart", "landing_page_view", "video_view",
            "post", "post_reaction", "initiate_checkout", "post_interaction_gross",
            "view_content", "post_engagement", "page_engagement",
        ]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Optional: sort for readability
        sort_columns = ["campaign_name", "date", "hourly_window"]
        available_columns = [col for col in sort_columns if col in df.columns]
        if available_columns:
            df = df.sort_values(available_columns).reset_index(drop=True)

        return df

    def fetch_and_save(
        self,
        ad_account_id: str = None,
        access_token: str = None,
        since: str = None,
        until: str = None,
        output_path: str = None
    ):
        """Fetch data and save to CSV file."""
        # Use config defaults if not provided
        ad_account_id = ad_account_id or os.getenv("AD_ACCOUNT_ID", "act_1685189008684458")
        access_token = "EAAJ1b4rDAIQBO4WmTAhBdhLyTCZCfBVvOr2iAbYMLVWpIKfZAP2bWVp49ucZBzlRTpud5ZCRZB7Ae6pcEleUrZCgfER9enRiPQdCFcGW6TRHXtfcfNO5IVFBZCOj1HDZBaIp8EFJL2ZAZCABXYVrV5Qg7lWZAjuVuMfFX1uZBfPUPATvJ0OwY5w7tcoXqmK6LNwZBtQZDZD"
        since = "2025-10-01"
        until = "2025-10-20"
        output_path = output_path or config.get("data.raw_dir", "data/raw") + "/meta_insights_hourly_dataset.csv"

        if not access_token or access_token.startswith("<PUT_"):
            raise ValueError("Please set ACCESS_TOKEN env var or provide valid access_token.")

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Fetching hourly insights...")
        raw = self.fetch_insights_hourly(
            ad_account_id=ad_account_id,
            access_token=access_token,
            since=since,
            until=until,
            level="campaign",
            limit=1000,
            sleep_sec=0.5,
            max_retries=5,
            rate_limit_delay=2.0,
        )
        self.logger.info(f"Fetched {len(raw)} rows.")

        self.logger.info("Building dataset...")
        df = self.build_dataset(raw)

        self.logger.info(f"Writing CSV → {output_path}")
        df.to_csv(output_path, index=False)
        self.logger.info("Done.")

        return df


# Convenience function for direct usage
def fetch_meta_insights(**kwargs):
    """Fetch Meta Ads insights data."""
    fetcher = MetaInsightsFetcher()
    return fetcher.fetch_and_save(**kwargs)


if __name__ == "__main__":
    fetch_meta_insights()
