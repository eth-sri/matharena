#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime


OAI_BASE = "https://export.arxiv.org/oai2"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}


def fetch_xml(url):
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def iter_records(from_date, until_date, set_spec="math"):
    params = {
        "verb": "ListRecords",
        "metadataPrefix": "arXiv",
        "set": set_spec,
        "from": from_date,
        "until": until_date,
    }
    resumption_token = None
    while True:
        if resumption_token:
            params = {"verb": "ListRecords", "resumptionToken": resumption_token}
        url = OAI_BASE + "?" + urllib.parse.urlencode(params)
        data = fetch_xml(url)
        root = ET.fromstring(data)
        for record in root.findall(".//oai:record", NS):
            yield record
        token_el = root.find(".//oai:resumptionToken", NS)
        if token_el is None or not (token_el.text or "").strip():
            break
        resumption_token = token_el.text.strip()


def record_to_metadata(record):
    header = record.find("oai:header", NS)
    if header is not None and header.get("status") == "deleted":
        return None
    meta = record.find("oai:metadata/arxiv:arXiv", NS)
    if meta is None:
        return None

    def text(path):
        el = meta.find(path, NS)
        if el is None or el.text is None:
            return None
        return " ".join(el.text.split())

    arxiv_id = text("arxiv:id")
    authors = []
    for author in meta.findall("arxiv:authors/arxiv:author", NS):
        authors.append(
            {
                "keyname": (author.findtext("arxiv:keyname", default="", namespaces=NS) or "").strip(),
                "forenames": (author.findtext("arxiv:forenames", default="", namespaces=NS) or "").strip(),
            }
        )

    categories_raw = text("arxiv:categories") or ""
    metadata = {
        "id": arxiv_id,
        "created": text("arxiv:created"),
        "updated": text("arxiv:updated"),
        "title": text("arxiv:title"),
        "abstract": text("arxiv:abstract"),
        "categories": [c for c in categories_raw.split() if c],
        "comments": text("arxiv:comments"),
        "journal_ref": text("arxiv:journal-ref"),
        "doi": text("arxiv:doi"),
        "license": text("arxiv:license"),
        "authors": authors,
    }
    if header is not None:
        metadata["oai_identifier"] = header.findtext("oai:identifier", default="", namespaces=NS).strip()
        metadata["datestamp"] = header.findtext("oai:datestamp", default="", namespaces=NS).strip()
    return metadata


def safe_dir_name(arxiv_id):
    return arxiv_id.replace("/", "_")


def write_metadata(record, metadata, out_dir):
    json_path = os.path.join(out_dir, "metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def allowed_yymm_prefixes(from_date, until_date):
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(until_date, "%Y-%m-%d")
    prefixes = set()
    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        prefixes.add(f"{year % 100:02d}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return prefixes


def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv math metadata in a date range."
    )
    parser.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--until", dest="until_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--outdir", default="arxivmath/paper", help="Output root directory")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between records")
    parser.add_argument("--skip-existing", action="store_true", help="Skip papers already downloaded")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    total = 0
    skipped = 0
    skipped_prefix = 0
    allowed_prefixes = allowed_yymm_prefixes(args.from_date, args.until_date)
    for record in iter_records(args.from_date, args.until_date):
        metadata = record_to_metadata(record)
        if metadata is None or not metadata.get("id"):
            continue
        arxiv_id = metadata["id"]
        if len(arxiv_id) >= 4 and arxiv_id[0:4].isdigit():
            if arxiv_id[0:4] not in allowed_prefixes:
                skipped_prefix += 1
                continue
        paper_dir = os.path.join(args.outdir, safe_dir_name(arxiv_id))
        if args.skip_existing and os.path.isdir(paper_dir):
            skipped += 1
            continue
        ensure_dir(paper_dir)
        write_metadata(record, metadata, paper_dir)
        total += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(
        f"Downloaded metadata for {total} papers. Skipped {skipped} existing, "
        f"{skipped_prefix} outside yymm."
    )


if __name__ == "__main__":
    sys.exit(main())
