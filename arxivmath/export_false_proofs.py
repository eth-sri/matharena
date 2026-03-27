#!/usr/bin/env python3
import argparse
import csv
import json
import os

from matharena.arxivbench_utils import get_latest_fields, list_paper_ids, load_annotation


ANNOTATION_FILENAME = "llm_metadata_false.json"


def load_metadata(paper_root, paper_id):
    with open(os.path.join(paper_root, paper_id, "metadata.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def is_accepted(annotation):
    if not annotation:
        return False
    review = annotation.get("review") or {}
    if review.get("status") != "keep":
        return False
    if annotation.get("keep") is not True:
        return False
    verification = annotation.get("verification") or {}
    if verification.get("keep") is not True:
        return False
    prior_work = annotation.get("prior_work_filter") or {}
    if prior_work:
        parsed = prior_work.get("parsed") or {}
        if parsed.get("action") != "keep":
            return False
    solid_authors = annotation.get("solid_authors") or {}
    if solid_authors:
        parsed = solid_authors.get("parsed") or {}
        if parsed.get("keep") is not True:
            return False
    return True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Export accepted false arXiv benchmark items as a proof competition.")
    parser.add_argument("--paper-root", default="arxivmath/paper")
    parser.add_argument("--out-dir", default="data/arxiv_false/february")
    args = parser.parse_args()

    accepted = []
    for paper_id in list_paper_ids(args.paper_root):
        annotation = load_annotation(args.paper_root, paper_id, ANNOTATION_FILENAME)
        if not is_accepted(annotation):
            continue
        fields = get_latest_fields(annotation, ["original_statement", "perturbed_statement", "falsity_explanation"])
        if not fields:
            continue
        original_statement, perturbed_statement, falsity_explanation = fields
        accepted.append((paper_id, original_statement, perturbed_statement, load_metadata(args.paper_root, paper_id)))

    if not accepted:
        print("No accepted papers found.")
        return

    problems_dir = os.path.join(args.out_dir, "problems")
    original_dir = os.path.join(args.out_dir, "original")
    ensure_dir(problems_dir)
    ensure_dir(original_dir)

    source_path = os.path.join(args.out_dir, "source.csv")
    source_meta_path = os.path.join(args.out_dir, "source_metadata.csv")
    types_path = os.path.join(args.out_dir, "problem_types.csv")
    grading_path = os.path.join(args.out_dir, "grading_scheme.json")

    grading_scheme = []
    with open(source_path, "w", encoding="utf-8", newline="") as source_file, open(
        source_meta_path, "w", encoding="utf-8", newline=""
    ) as source_meta_file, open(types_path, "w", encoding="utf-8", newline="") as types_file:
        source_writer = csv.writer(source_file, lineterminator="\n")
        source_meta_writer = csv.writer(source_meta_file, lineterminator="\n")
        types_writer = csv.writer(types_file, lineterminator="\n")
        source_writer.writerow(["id", "source"])
        source_meta_writer.writerow(["id", "title", "authors"])
        types_writer.writerow(["id", "type"])

        for idx, (paper_id, original_statement, statement, metadata) in enumerate(accepted, start=1):
            write_text(os.path.join(problems_dir, f"{idx}.tex"), statement)
            write_text(os.path.join(original_dir, f"{idx}.tex"), original_statement)
            grading_scheme.append(
                {
                    "id": str(idx),
                    "points": 2,
                    "scheme": "",
                    "ground_truth_proofs": [],
                }
            )
            title = metadata.get("title") or ""
            author_names = []
            for author in metadata.get("authors") or []:
                full_name = " ".join([author.get("forenames", "").strip(), author.get("keyname", "").strip()]).strip()
                if full_name:
                    author_names.append(full_name)
            source_writer.writerow([idx, paper_id])
            source_meta_writer.writerow([idx, title, "; ".join(author_names)])
            types_writer.writerow([idx, "[]"])

    with open(grading_path, "w", encoding="utf-8") as f:
        json.dump(grading_scheme, f, indent=2)

    print(f"Exported {len(accepted)} false-proof problems to {args.out_dir}")


if __name__ == "__main__":
    main()
