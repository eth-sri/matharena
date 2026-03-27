#!/usr/bin/env python3
import argparse
import csv
import os
import json

from matharena.arxivbench_utils import load_annotation, list_paper_ids


def load_metadata(paper_root, paper_id):
    path = os.path.join(paper_root, paper_id, "metadata.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
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
    return verification.get("keep") is True


def get_reviewed_pair(annotation):
    review = annotation.get("review") or {}
    question = review.get("question")
    answer = review.get("answer")
    if not question or not answer:
        return None, None
    return question.strip(), answer.strip()


def list_paper_ids(paper_root):
    if not os.path.isdir(paper_root):
        return []
    paper_ids = []
    for name in os.listdir(paper_root):
        path = os.path.join(paper_root, name)
        if os.path.isdir(path):
            paper_ids.append(name)
    return sorted(paper_ids)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export accepted arXiv questions into data/arxiv/december."
    )
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--out-dir", default="data/arxiv/january", help="Output dataset directory.")
    args = parser.parse_args()

    paper_ids = list_paper_ids(args.paper_root)
    accepted = []
    skipped_missing_review = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id)
        if not is_accepted(annotation):
            continue
        question, answer = get_reviewed_pair(annotation)
        if not question or not answer:
            skipped_missing_review.append(paper_id)
            continue
        metadata = load_metadata(args.paper_root, paper_id)
        if not metadata:
            skipped_missing_review.append(paper_id)
            continue
        accepted.append((paper_id, question, answer, metadata))

    if not accepted:
        print("No accepted papers found.")
        return

    problems_dir = os.path.join(args.out_dir, "problems")
    ensure_dir(problems_dir)

    answers_path = os.path.join(args.out_dir, "answers.csv")
    source_path = os.path.join(args.out_dir, "source.csv")
    source_meta_path = os.path.join(args.out_dir, "source_metadata.csv")
    types_path = os.path.join(args.out_dir, "problem_types.csv")
    with open(answers_path, "w", encoding="utf-8", newline="") as answers_file, open(
        source_path, "w", encoding="utf-8", newline=""
    ) as source_file, open(source_meta_path, "w", encoding="utf-8", newline="") as source_meta_file, open(
        types_path, "w", encoding="utf-8", newline=""
    ) as types_file:
        answers_writer = csv.writer(answers_file, lineterminator="\n")
        source_writer = csv.writer(source_file, lineterminator="\n")
        source_meta_writer = csv.writer(source_meta_file, lineterminator="\n")
        types_writer = csv.writer(types_file, lineterminator="\n")
        answers_writer.writerow(["id", "answer"])
        source_writer.writerow(["id", "source"])
        source_meta_writer.writerow(["id", "title", "authors"])
        types_writer.writerow(["id", "type"])
        for idx, (paper_id, question, answer, metadata) in enumerate(accepted, start=1):
            write_text(os.path.join(problems_dir, f"{idx}.tex"), question)
            answers_writer.writerow([idx, answer])
            types_writer.writerow([idx, "[]"])
            title = metadata.get("title") or ""
            authors = metadata.get("authors") or []
            author_names = []
            for author in authors:
                forenames = (author.get("forenames") or "").strip()
                keyname = (author.get("keyname") or "").strip()
                full_name = " ".join([p for p in [forenames, keyname] if p])
                if full_name:
                    author_names.append(full_name)
            source_writer.writerow([idx, paper_id])
            source_meta_writer.writerow([idx, title, "; ".join(author_names)])

    print(f"Exported {len(accepted)} questions to {args.out_dir}")
    if skipped_missing_review:
        print(
            "Skipped missing reviewed question/answer: "
            + ", ".join(sorted(skipped_missing_review))
        )


if __name__ == "__main__":
    main()
