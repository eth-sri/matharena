#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime

from matharena.arxivbench_utils import list_paper_ids, load_annotation, load_metadata


ANNOTATION_FILENAME = "metadata_lean_abstract.json"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def latest_value(annotation, key):
    review = annotation.get("review") or {}
    return (review.get(key) or annotation.get(key) or "").strip()


def is_accepted(annotation):
    if not annotation:
        return False
    review = annotation.get("review") or {}
    if "review" in annotation and review.get("status") != "keep" and review:
        return False
    if not annotation.get("keep"):
        return False
    # verification = annotation.get("verification") or {}
    # if verification.get("keep") is not True:
    #     return False
    # semantic = annotation.get("semantic_verification") or {}
    # if semantic.get("keep") is not True:
    #     return False
    return bool(latest_value(annotation, "statement") and latest_value(annotation, "formalized_statement"))


def main():
    parser = argparse.ArgumentParser(description="Export accepted abstract-derived Lean benchmark items.")
    parser.add_argument("--paper-root", default="arxivmath/paper")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--annotation-filename", default=ANNOTATION_FILENAME)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    accepted = []
    for paper_id in list_paper_ids(args.paper_root):
        annotation = load_annotation(args.paper_root, paper_id, args.annotation_filename)
        if not is_accepted(annotation):
            continue
        metadata = load_metadata(args.paper_root, paper_id)
        accepted.append(
            (
                paper_id,
                latest_value(annotation, "statement"),
                latest_value(annotation, "formalized_statement"),
                (metadata.get("abstract") or "").strip(),
                metadata,
            )
        )
        if args.limit is not None and len(accepted) >= args.limit:
            break

    if not accepted:
        print("No accepted abstract-derived papers found.")
        return

    problems_dir = os.path.join(args.out_dir, "problems")
    original_dir = os.path.join(args.out_dir, "original")
    for path in (problems_dir, original_dir):
        ensure_dir(path)

    source_path = os.path.join(args.out_dir, "source.csv")
    source_meta_path = os.path.join(args.out_dir, "source_metadata.csv")

    with open(source_path, "w", encoding="utf-8", newline="") as source_file, open(
        source_meta_path, "w", encoding="utf-8", newline=""
    ) as source_meta_file:
        source_writer = csv.writer(source_file, lineterminator="\n")
        source_meta_writer = csv.writer(source_meta_file, lineterminator="\n")
        source_writer.writerow(["id", "source"])
        source_meta_writer.writerow(["id", "title", "authors"])

        for idx, (paper_id, statement, formalized_statement, abstract, metadata) in enumerate(accepted, start=1):
            write_text(os.path.join(problems_dir, f"{idx}.lean"), formalized_statement)
            write_text(os.path.join(original_dir, f"{idx}.tex"), statement)

            title = metadata.get("title") or ""
            author_names = []
            for author in metadata.get("authors") or []:
                full_name = " ".join([author.get("forenames", "").strip(), author.get("keyname", "").strip()]).strip()
                if full_name:
                    author_names.append(full_name)
            source_writer.writerow([idx, paper_id])
            source_meta_writer.writerow([idx, title, "; ".join(author_names)])

    print(f"Exported {len(accepted)} abstract-derived Lean formalization problems to {args.out_dir}")


if __name__ == "__main__":
    main()
