#!/usr/bin/env python3
"""
Script to remove or reorder problems by ID from any competition.

For removing problems:
1. Take problem IDs to remove
2. Remove those problems from answers.csv or grading_scheme.json and source.csv (if it exists),
   and source_metadata.csv (if it exists)
3. Renumber remaining problems to maintain sequential IDs
4. Remove corresponding .tex files from problems directory
5. Remove corresponding .json files from outputs subdirectories
6. Update idx fields in remaining JSON files

For reordering problems:
1. Take a permutation of problem IDs
2. Reorder problems in answers.csv or grading_scheme.json, source.csv, and source_metadata.csv (if it exists)
3. Rename .tex files in problems directory according to the new order
4. Rename .json files in outputs subdirectories and update their idx field
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set


def parse_problem_ids(problem_ids: List[str]) -> Set[int]:
    """Convert string IDs to integer IDs."""
    ids_to_remove = set()
    
    for id_str in problem_ids:
        try:
            problem_id = int(id_str)
            ids_to_remove.add(problem_id)
        except ValueError:
            print(f"Error: '{id_str}' is not a valid integer ID")
    
    return ids_to_remove

def parse_permutation(perm_str: List[str]) -> List[int]:
    """Convert string permutation to integer list."""
    permutation = []
    for id_str in perm_str:
        try:
            problem_id = int(id_str)
            permutation.append(problem_id)
        except ValueError:
            print(f"Error: '{id_str}' is not a valid integer ID for a permutation.")
            return []
    return permutation

def reorder_csv_files(source_csv_path: str, source_metadata_csv_path: str, answers_csv_path: str,
                      permutation: List[int], has_source_csv: bool, has_source_metadata_csv: bool,
                      use_grading_scheme: bool = False, grading_scheme_path: str = None):
    """Reorder metadata plus answers.csv or grading_scheme.json based on a permutation."""
    source_rows_map = {}
    if has_source_csv:
        with open(source_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_rows_map[int(row['id'])] = row

    source_metadata_rows_map = {}
    source_metadata_fieldnames = None
    if has_source_metadata_csv:
        with open(source_metadata_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            source_metadata_fieldnames = reader.fieldnames
            for row in reader:
                source_metadata_rows_map[int(row['id'])] = row

    answers_rows_map = {}
    grading_scheme_map = {}
    if use_grading_scheme:
        if not grading_scheme_path:
            print("Error: grading_scheme_path is required when use_grading_scheme=True.")
            return False
        with open(grading_scheme_path, 'r', encoding='utf-8') as f:
            grading_scheme = json.load(f)
            for row in grading_scheme:
                grading_scheme_map[int(row['id'])] = row
        num_problems = len(grading_scheme_map)
    else:
        with open(answers_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                answers_rows_map[int(row['id'])] = row
        num_problems = len(answers_rows_map)

    if len(permutation) != num_problems:
        print(f"Error: Permutation length ({len(permutation)}) does not match number of problems ({num_problems}).")
        return False
    if sorted(permutation) != list(range(1, num_problems + 1)):
        print(f"Error: Permutation must be a rearrangement of numbers from 1 to {num_problems}.")
        return False

    new_source_rows = []
    new_source_metadata_rows = []
    new_answers_rows = []
    new_grading_scheme_rows = []

    for new_id, old_id in enumerate(permutation, 1):
        if use_grading_scheme:
            if old_id in grading_scheme_map:
                new_grading_scheme_rows.append(grading_scheme_map[old_id])
                if has_source_csv and old_id in source_rows_map:
                    new_source_rows.append(source_rows_map[old_id])
                if has_source_metadata_csv and old_id in source_metadata_rows_map:
                    new_source_metadata_rows.append(source_metadata_rows_map[old_id])
        else:
            if old_id in answers_rows_map:
                new_answers_rows.append(answers_rows_map[old_id])
                if has_source_csv and old_id in source_rows_map:
                    new_source_rows.append(source_rows_map[old_id])
                if has_source_metadata_csv and old_id in source_metadata_rows_map:
                    new_source_metadata_rows.append(source_metadata_rows_map[old_id])

    for i, row in enumerate(new_source_rows, 1):
        row['id'] = str(i)
    for i, row in enumerate(new_source_metadata_rows, 1):
        row['id'] = str(i)
    for i, row in enumerate(new_answers_rows, 1):
        row['id'] = str(i)
    for i, row in enumerate(new_grading_scheme_rows, 1):
        row['id'] = str(i)

    if has_source_csv:
        with open(source_csv_path, 'w', encoding='utf-8', newline='') as f:
            if new_source_rows:
                writer = csv.DictWriter(f, fieldnames=['id', 'source'])
                writer.writeheader()
                writer.writerows(new_source_rows)

    if has_source_metadata_csv:
        fieldnames = source_metadata_fieldnames or ['id', 'title', 'authors']
        if 'id' not in fieldnames:
            fieldnames = ['id'] + fieldnames
        with open(source_metadata_csv_path, 'w', encoding='utf-8', newline='') as f:
            if new_source_metadata_rows:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(new_source_metadata_rows)

    if use_grading_scheme:
        with open(grading_scheme_path, 'w', encoding='utf-8') as f:
            json.dump(new_grading_scheme_rows, f, indent=4, ensure_ascii=False)
    else:
        with open(answers_csv_path, 'w', encoding='utf-8', newline='') as f:
            if new_answers_rows:
                writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
                writer.writeheader()
                writer.writerows(new_answers_rows)
    
    return True


def reorder_files(directory: str, permutation: List[int], extension: str, update_json_content: bool = False):
    """Reorder files in a directory based on a permutation."""
    dir_path = Path(directory)
    if not dir_path.exists():
        # Silently ignore non-existing directories, like 'problems'
        return

    old_id_to_new_id = {old_id: new_id for new_id, old_id in enumerate(permutation, 1)}

    # Rename to temp files
    temp_files = []
    processed_old_ids = set()
    for old_id, new_id in old_id_to_new_id.items():
        old_file = dir_path / f"{old_id}{extension}"
        if old_file.exists():
            temp_name = dir_path / f"temp_{new_id}{extension}"
            old_file.rename(temp_name)
            temp_files.append(temp_name)
            processed_old_ids.add(old_id)

            if update_json_content:
                try:
                    with open(temp_name, 'r+', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'idx' in data:
                            data['idx'] = new_id
                            f.seek(0)
                            json.dump(data, f, indent=4, ensure_ascii=False)
                            f.truncate()
                    print(f"  Updated idx in temp file for new problem {new_id}")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"  Error updating {old_file}: {e}")

    # Rename from temp to final
    for temp_file in temp_files:
        new_id_str = temp_file.stem.split('_')[1]
        final_name = dir_path / f"{new_id_str}{extension}"
        temp_file.rename(final_name)
        print(f"Renamed file to {final_name.name}")

def reorder_problems_directory(problems_dir: str, permutation: List[int]):
    """Reorder .tex files based on a permutation."""
    reorder_files(problems_dir, permutation, ".tex")


def reorder_json_outputs(outputs_dir: str, permutation: List[int]):
    """Reorder JSON files and update their idx field."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        print(f"Outputs directory {outputs_dir} does not exist")
        return

    for subdir in outputs_path.rglob("*"):
        if subdir.is_dir() and any(subdir.glob("*.json")):
            print(f"Processing JSON files in {subdir}")
            reorder_files(str(subdir), permutation, ".json", update_json_content=True)

def update_csv_files(source_csv_path: str, source_metadata_csv_path: str, answers_csv_path: str,
                     ids_to_remove: Set[int], has_source_csv: bool, has_source_metadata_csv: bool,
                     source_csv_to: str = None, source_metadata_csv_to: str = None,
                     answers_csv_to: str = None, use_grading_scheme: bool = False,
                     grading_scheme_path: str = None, grading_scheme_to: str = None) -> int:
    """Remove problems from metadata plus answers.csv or grading_scheme.json and renumber remaining ones."""
    
    source_rows = []
    source_rows_to = []
    if has_source_csv:
        # Read source.csv
        with open(source_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['id']) not in ids_to_remove:
                    source_rows.append(row)
                else:
                    source_rows_to.append(row)

    source_metadata_rows = []
    source_metadata_rows_to = []
    source_metadata_fieldnames = None
    if has_source_metadata_csv:
        with open(source_metadata_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            source_metadata_fieldnames = reader.fieldnames
            for row in reader:
                if int(row['id']) not in ids_to_remove:
                    source_metadata_rows.append(row)
                else:
                    source_metadata_rows_to.append(row)
    
    answers_rows = []
    answers_rows_to = []
    grading_scheme_rows: List[Dict[str, Any]] = []
    grading_scheme_rows_to: List[Dict[str, Any]] = []
    if use_grading_scheme:
        if not grading_scheme_path:
            raise ValueError("grading_scheme_path is required when use_grading_scheme=True.")
        with open(grading_scheme_path, 'r', encoding='utf-8') as f:
            grading_scheme = json.load(f)
            for row in grading_scheme:
                if int(row['id']) not in ids_to_remove:
                    grading_scheme_rows.append(row)
                else:
                    grading_scheme_rows_to.append(row)
    else:
        with open(answers_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['id']) not in ids_to_remove:
                    answers_rows.append(row)
                else:
                    answers_rows_to.append(row)
    
    # Renumber remaining rows
    if has_source_csv:
        for i, row in enumerate(source_rows, 1):
            row['id'] = str(i)

    if has_source_metadata_csv:
        for i, row in enumerate(source_metadata_rows, 1):
            row['id'] = str(i)
    
    for i, row in enumerate(answers_rows, 1):
        row['id'] = str(i)
    for i, row in enumerate(grading_scheme_rows, 1):
        row['id'] = str(i)
    
    # Write back source.csv only if it exists
    if has_source_csv:
        with open(source_csv_path, 'w', encoding='utf-8', newline='') as f:
            if source_rows:
                writer = csv.DictWriter(f, fieldnames=['id', 'source'])
                writer.writeheader()
                writer.writerows(source_rows)

    if has_source_metadata_csv:
        fieldnames = source_metadata_fieldnames or ['id', 'title', 'authors']
        if 'id' not in fieldnames:
            fieldnames = ['id'] + fieldnames
        with open(source_metadata_csv_path, 'w', encoding='utf-8', newline='') as f:
            if source_metadata_rows:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(source_metadata_rows)
    
    if source_csv_to and os.path.exists(source_csv_to):
        # get current length of target file to append correctly
        current_length = 0
        with open(source_csv_to, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
            current_length = sum(1 for line in lines if len(line.strip()) > 0)
            empty_last_line = "\n" in lines[-1]
        with open(source_csv_to, 'a', encoding='utf-8', newline="") as f:
            if not empty_last_line:
                f.write('\n')
            writer = csv.DictWriter(f, fieldnames=['id', 'source'])
            if f.tell() == 0:
                writer.writeheader()
            for i, row in enumerate(source_rows_to):
                row['id'] = str(i + current_length)
            writer.writerows(source_rows_to)

    if source_metadata_csv_to and os.path.exists(source_metadata_csv_to):
        current_length = 0
        with open(source_metadata_csv_to, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
            current_length = sum(1 for line in lines if len(line.strip()) > 0)
            empty_last_line = "\n" in lines[-1]
        with open(source_metadata_csv_to, 'a', encoding='utf-8', newline="") as f:
            if not empty_last_line:
                f.write('\n')
            fieldnames = source_metadata_fieldnames or ['id', 'title', 'authors']
            if 'id' not in fieldnames:
                fieldnames = ['id'] + fieldnames
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            for i, row in enumerate(source_metadata_rows_to):
                row['id'] = str(i + current_length)
            writer.writerows(source_metadata_rows_to)
    
    if use_grading_scheme:
        with open(grading_scheme_path, 'w', encoding='utf-8') as f:
            json.dump(grading_scheme_rows, f, indent=4, ensure_ascii=False)

        if grading_scheme_to:
            target_rows = []
            if os.path.exists(grading_scheme_to):
                with open(grading_scheme_to, 'r', encoding='utf-8') as f:
                    target_rows = json.load(f)
            current_length = len(target_rows)
            for i, row in enumerate(grading_scheme_rows_to, 1):
                row['id'] = str(current_length + i)
            target_rows.extend(grading_scheme_rows_to)
            with open(grading_scheme_to, 'w', encoding='utf-8') as f:
                json.dump(target_rows, f, indent=4, ensure_ascii=False)
        return len(grading_scheme_rows)

    # Write back answers.csv
    with open(answers_csv_path, 'w', encoding='utf-8', newline='') as f:
        if answers_rows:
            writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
            writer.writeheader()
            writer.writerows(answers_rows)

    if answers_csv_to:
        existing_rows = []
        if os.path.exists(answers_csv_to):
            with open(answers_csv_to, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        else:
            Path(answers_csv_to).parent.mkdir(parents=True, exist_ok=True)

        current_length = len(existing_rows)
        for i, row in enumerate(answers_rows_to, 1):
            row['id'] = str(current_length + i)
        all_rows = existing_rows + answers_rows_to

        with open(answers_csv_to, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
            writer.writeheader()
            writer.writerows(all_rows)

    return len(answers_rows)

def update_problems_directory(problems_dir: str, ids_to_remove: Set[int], final_count: int, 
                              problems_dir_to: str = None, n_to_problems: int = None):
    """Remove .tex files and renumber remaining ones."""
    problems_path = Path(problems_dir)
    
    if not problems_path.exists():
        print(f"Problems directory {problems_dir} does not exist")
        return
    
    # Remove files for deleted problems
    problems = []
    for problem_id in ids_to_remove:
        tex_file = problems_path / f"{problem_id}.tex"
        if tex_file.exists():
            problems.append(str(open(tex_file, 'r', encoding='utf-8').read()))
            tex_file.unlink()
            print(f"Removed {tex_file}")
        png_file = problems_path / f"{problem_id}.png"
        if png_file.exists():
            png_file.unlink()
            print(f"Removed {png_file}")
    
    # Create mapping of old IDs to new IDs
    existing_files = []
    for tex_file in problems_path.glob("*.tex"):
        old_id = int(tex_file.stem)
        if old_id not in ids_to_remove:
            existing_files.append((old_id, tex_file))
    
    # Sort by old ID to maintain order
    existing_files.sort(key=lambda x: x[0])
    
    # Rename files to temporary names first to avoid conflicts
    temp_files = []
    for new_id, (old_id, old_file) in enumerate(existing_files, 1):
        temp_name = problems_path / f"temp_{new_id}.tex"
        old_file.rename(temp_name)
        temp_files.append((new_id, temp_name))
        old_file_png = problems_path / f"{old_id}.png"
        if old_file_png.exists():
            temp_name_png = problems_path / f"temp_{new_id}.png"
            old_file_png.rename(temp_name_png)
            print(f"Renamed {old_file_png} to {temp_name_png}")
    
    # Rename from temporary names to final names
    for new_id, temp_file in temp_files:
        final_file = problems_path / f"{new_id}.tex"
        temp_file.rename(final_file)
        print(f"Renamed problem file to {new_id}.tex")
        temp_file_png = problems_path / f"temp_{new_id}.png"
        if temp_file_png.exists():
            final_file_png = problems_path / f"{new_id}.png"
            temp_file_png.rename(final_file_png)
            print(f"Renamed image file to {new_id}.png")

    if problems_dir_to:
        problems_path_to = Path(problems_dir_to)
        for i, content in enumerate(problems, n_to_problems + 1):
            target_file = problems_path_to / f"{i}.tex"
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Moved removed problem to {target_file}")

def update_json_outputs(outputs_dir: str, ids_to_remove: Set[int], 
                        outputs_dir_to: str = None, n_to_problems: int = None):
    """Remove JSON files and renumber remaining ones, updating idx fields."""
    outputs_path = Path(outputs_dir)
    sorted_ids_to_remove = sorted(list(ids_to_remove))
    
    if not outputs_path.exists():
        print(f"Outputs directory {outputs_dir} does not exist")
        return
    
    # Find all subdirectories with JSON files
    for subdir in outputs_path.rglob("*"):
        if subdir.is_dir() and any(subdir.glob("*.json")):
            print(f"Processing JSON files in {subdir}")
            
            # Remove files for deleted problems
            for problem_id in ids_to_remove:
                json_file = subdir / f"{problem_id}.json"
                if json_file.exists():
                    move_path = str(subdir).replace(str(outputs_path), str(outputs_dir_to)) if outputs_dir_to else None
                    move_filename = f"{n_to_problems + sorted_ids_to_remove.index(problem_id) + 1}.json" if outputs_dir_to else None
                    if move_path and move_filename:
                        target_dir = Path(move_path)
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_file = target_dir / move_filename
                        shutil.move(str(json_file), str(target_file))
                        with open(target_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            data["idx"] = n_to_problems + sorted_ids_to_remove.index(problem_id) + 1
                        with open(target_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        print(f"  Moved {json_file} to {target_file}")
                    else:
                        json_file.unlink()
                        print(f"  Removed {json_file}")
            
            # Get existing JSON files and their IDs
            existing_files = []
            for json_file in subdir.glob("*.json"):
                try:
                    old_id = int(json_file.stem)
                    if old_id not in ids_to_remove:
                        existing_files.append((old_id, json_file))
                except ValueError:
                    # Skip files that don't have numeric names
                    continue
            
            # Sort by old ID to maintain order
            existing_files.sort(key=lambda x: x[0])
            
            # Process each file: update idx and rename
            temp_files = []
            for new_id, (old_id, old_file) in enumerate(existing_files, 1):
                # Read and update JSON content
                try:
                    with open(old_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Update idx field if it exists
                    if isinstance(data, dict) and 'idx' in data:
                        data['idx'] = new_id
                    
                    # Write to temporary file
                    temp_name = subdir / f"temp_{new_id}.json"
                    with open(temp_name, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    
                    temp_files.append((new_id, temp_name, old_file))
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"  Error processing {old_file}: {e}")
                    continue
            
            # Remove old files and rename temp files
            for new_id, temp_file, old_file in temp_files:
                old_file.unlink()  # Remove old file
                final_file = subdir / f"{new_id}.json"
                temp_file.rename(final_file)  # Rename temp to final
                print(f"  Updated and renamed to {new_id}.json")

def main():
    parser = argparse.ArgumentParser(description="Remove or reorder problems in a competition dataset.")
    parser.add_argument("competition", help="Competition name (e.g., 'apex/apex', 'aime/aime_2024_I')")
    parser.add_argument("problem_ids", nargs='+', help="For removing: problem IDs to remove. For reordering: a permutation of all problem IDs.")
    parser.add_argument("--reorder", action="store_true", help="Enable reordering mode. problem_ids will be treated as a permutation.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--to-competition", default=None, help="Target competition for moving problems")

    args = parser.parse_args()
    
    # Define paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Go up from scripts/curation/ to project root
    data_dir = project_root / "data" / args.competition
    source_csv = data_dir / "source.csv"
    source_metadata_csv = data_dir / "source_metadata.csv"
    answers_csv = data_dir / "answers.csv"
    grading_scheme_json = data_dir / "grading_scheme.json"
    problems_dir = data_dir / "problems"
    outputs_dir = project_root / "outputs" / args.competition

    data_dir_to = None
    source_csv_to = None
    source_metadata_csv_to = None
    answers_csv_to = None
    grading_scheme_json_to = None
    problems_dir_to = None
    outputs_dir_to = None
    if args.to_competition:
        data_dir_to = project_root / "data" / args.to_competition
        source_csv_to = data_dir_to / "source.csv"
        source_metadata_csv_to = data_dir_to / "source_metadata.csv"
        answers_csv_to = data_dir_to / "answers.csv"
        grading_scheme_json_to = data_dir_to / "grading_scheme.json"
        problems_dir_to = data_dir_to / "problems"
        outputs_dir_to = project_root / "outputs" / args.to_competition
    
    has_source_csv = source_csv.exists()
    has_source_metadata_csv = source_metadata_csv.exists()
    has_answers_csv = answers_csv.exists()
    has_grading_scheme_json = grading_scheme_json.exists()
    use_grading_scheme = (not has_answers_csv) and has_grading_scheme_json

    if not has_answers_csv and not has_grading_scheme_json:
        print("Error: Neither answers.csv nor grading_scheme.json found.")
        return
    if has_answers_csv and has_grading_scheme_json:
        print("Warning: Both answers.csv and grading_scheme.json found; defaulting to answers.csv.")

    print(f"Working with competition: {args.competition}")

    if args.reorder:
        permutation = parse_permutation(args.problem_ids)
        if not permutation:
            return

        if args.dry_run:
            print("\nDRY RUN - No changes will be made")
            print(f"Would reorder problems with permutation: {permutation}")
            print(f"Data directory: {data_dir}")
            print(f"Outputs directory: {outputs_dir}")
            return

        print(f"\nReordering problems with permutation: {permutation}")
        
        if not reorder_csv_files(
            str(source_csv),
            str(source_metadata_csv),
            str(answers_csv),
            permutation,
            has_source_csv,
            has_source_metadata_csv,
            use_grading_scheme=use_grading_scheme,
            grading_scheme_path=str(grading_scheme_json),
        ):
            print("Aborting due to dataset reordering error.")
            return
        print("Reordered dataset files.")

        reorder_problems_directory(str(problems_dir), permutation)
        reorder_json_outputs(str(outputs_dir), permutation)

        print(f"\nCompleted! Reordered {len(permutation)} problems.")

    else:
        # Nuke functionality
        print(f"Problem IDs to remove: {args.problem_ids}")
        ids_to_remove = parse_problem_ids(args.problem_ids)
        
        if not ids_to_remove:
            print("No valid problem IDs provided.")
            return
        
        try:
            existing_ids = set()
            if use_grading_scheme:
                with open(grading_scheme_json, 'r', encoding='utf-8') as f:
                    grading_rows = json.load(f)
                    for row in grading_rows:
                        existing_ids.add(int(row['id']))
            else:
                with open(answers_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing_ids.add(int(row['id']))
        except Exception:
            existing_ids = ids_to_remove
        
        missing_ids = ids_to_remove - existing_ids
        if missing_ids:
            print(f"Warning: The following IDs don't exist in the dataset: {sorted(missing_ids)}")
        
        ids_to_remove = ids_to_remove & existing_ids
        
        if not ids_to_remove:
            print("No valid problem IDs to remove.")
            return
        
        if args.dry_run:
            print("\nDRY RUN - No changes will be made")
            print(f"Would remove problem IDs: {sorted(ids_to_remove)}")
            print(f"Data directory: {data_dir}")
            print(f"Outputs directory: {outputs_dir}")
            return
        
        print(f"\nRemoving problem IDs: {sorted(ids_to_remove)}")
        
        final_count = len(existing_ids) - len(ids_to_remove)
        try:
            final_count = update_csv_files(
                str(source_csv),
                str(source_metadata_csv),
                str(answers_csv),
                ids_to_remove,
                has_source_csv,
                has_source_metadata_csv,
                str(source_csv_to) if source_csv_to else None,
                str(source_metadata_csv_to) if source_metadata_csv_to else None,
                str(answers_csv_to) if answers_csv_to else None,
                use_grading_scheme=use_grading_scheme,
                grading_scheme_path=str(grading_scheme_json),
                grading_scheme_to=str(grading_scheme_json_to) if grading_scheme_json_to else None,
            )
            print(f"Updated dataset files. {final_count} problems remaining.")
        except Exception as e:
            print(f"Error updating dataset files: {e}")
        
        
        n_to_problems = 0
        if problems_dir_to:
            n_to_problems = len(list(Path(problems_dir_to).glob("*.tex")))
        try:
            update_problems_directory(str(problems_dir), ids_to_remove, final_count, 
                                    str(problems_dir_to) if problems_dir_to else None, 
                                    n_to_problems)
        except Exception as e:
            print(f"Error updating problems directory: {e}")
        
        update_json_outputs(str(outputs_dir), ids_to_remove, 
                            str(outputs_dir_to) if outputs_dir_to else None,
                            n_to_problems)
        
        print(f"\nCompleted! Removed {len(ids_to_remove)} problems.")

if __name__ == "__main__":
    main()
