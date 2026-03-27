#!/usr/bin/env python3
"""
submit_euler_answers_playwright.py

A script to log into Project Euler and submit a list of answers for problem 946,
with a random delay between submissions, using Playwright, with optional CAPTCHA handling.

Requirements:
- playwright (pip install playwright)
- run `playwright install` to install browser binaries
"""
import argparse
import json
import csv
import glob
import time
import random
import os
import pickle
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
import subprocess
from matharena.json_zst import OUTPUT_JSON_SUFFIX, load_json_zst

load_dotenv()

# -- Configuration --
LOGIN_URL = "https://projecteuler.net/sign_in"
# Random delay (in seconds) between submissions:
MIN_DELAY = 20
MAX_DELAY = 60
# Cookie storage file
COOKIE_FILE = "scripts/euler/euler_cookies.pkl"


def convert_to_int(answer):
    try:
        answer = int(answer)
    except:
        answer = None
    return answer


def save_cookies(context):
    """Save cookies from the browser context to a file."""
    cookies = context.cookies()
    with open(COOKIE_FILE, "wb") as f:
        pickle.dump(cookies, f)
    print(f"Cookies saved to {COOKIE_FILE}")


def load_cookies():
    """Load cookies from file if it exists."""
    if os.path.exists(COOKIE_FILE):
        try:
            with open(COOKIE_FILE, "rb") as f:
                cookies = pickle.load(f)
            print(f"Loaded cookies from {COOKIE_FILE}")
            return cookies
        except Exception as e:
            print(f"Error loading cookies: {e}")
    return None


def is_logged_in(page):
    """Check if the user is logged in by looking for login indicators."""
    try:
        # Check for "Logged in as" text or logout link
        content = page.content()
        return "Logged in as" in content or "Sign out" in content
    except:
        return False


def solve_captcha(page):
    """
    Detects if a CAPTCHA is present on the page, prompts the user to solve it,
    and fills in the CAPTCHA field.
    """
    captcha_input = page.query_selector('input[name="captcha"]')
    if not captcha_input:
        return
    # Wait for CAPTCHA image to load
    try:
        page.wait_for_selector("#captcha_image[src]", timeout=5000)
    except PlaywrightTimeoutError:
        print("CAPTCHA image did not load in time.")
    # Get the CAPTCHA image URL
    captcha_src = page.get_attribute("#captcha_image", "src")
    print(f"Please solve the CAPTCHA displayed in your browser.\nImage URL: {captcha_src}")
    code = input("Enter CAPTCHA code: ")
    page.fill('input[name="captcha"]', code)


def login_with_cookies(page, context):
    """Try to login using saved cookies, fall back to manual login if needed."""
    # Try to load saved cookies
    cookies = load_cookies()

    if cookies:
        # Set cookies and navigate to the site
        context.add_cookies(cookies)
        page.goto("https://projecteuler.net/")
        page.wait_for_load_state("networkidle")

        # Check if we're already logged in
        if is_logged_in(page):
            print("Successfully logged in using saved cookies!")
            return True

    # If cookies didn't work or don't exist, do manual login
    print("Manual login required...")
    return False


def manual_login(page, username, password):
    """Perform manual login with credentials."""
    page.goto(LOGIN_URL)
    page.fill('input[name="username"]', username)
    page.fill('input[name="password"]', password)

    # Check if "Remember me" checkbox exists and click it
    try:
        remember_me_checkbox = page.locator('input[name="remember_me"]')
        if remember_me_checkbox.is_visible():
            remember_me_checkbox.check()
            print("Clicked 'Remember me' checkbox")
    except Exception as e:
        print(f"Could not find or click 'Remember me' checkbox: {e}")

    # Handle optional CAPTCHA
    solve_captcha(page)

    # Submit sign-in (covers both 'Login' and 'Sign In' buttons)
    page.click('input[name="sign_in"]')
    page.wait_for_load_state("networkidle")

    # Verify login
    if is_logged_in(page):
        print("Manual login successful.")
        return True
    else:
        print("Manual login failed. Check your credentials or CAPTCHA.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_id", type=int, required=True)
    parser.add_argument("--force-login", action="store_true", help="Force manual login even if cookies exist")
    parser.add_argument("--test", action="store_true", help="Use euler_test instead of euler")
    args = parser.parse_args()
    problem_id = args.problem_id

    euler = "euler_test" if args.test else "euler"

    euler_id = None
    with open(f"data/euler/{euler}/source.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == str(problem_id):
                euler_id = row["source"].replace("euler", "")
    if euler_id is None:
        raise ValueError(f"Problem ID {problem_id} not found in source.csv")

    # Find all files of the form outputs/euler/{euler}/{provider}/{model}/{problem_id}.json.zst
    files = glob.glob(f"outputs/euler/{euler}/*/*/{problem_id}{OUTPUT_JSON_SUFFIX}")
    if len(files) == 0:
        raise ValueError(f"No files found for problem ID {problem_id}")

    PROBLEM_URL = f"https://projecteuler.net/problem={euler_id}"

    all_answers = []
    for file in files:
        data = load_json_zst(file)
        print(file, data.get("answers", []))
        all_answers.extend(data.get("answers", []))

    # Filter out non-integer answers
    all_answers = [convert_to_int(answer) for answer in all_answers]
    all_answers = [answer for answer in all_answers if answer is not None]

    # Count occurrences of each answer
    from collections import Counter

    answer_counts = Counter(all_answers)

    # Sort by count decreasing and keep only unique elements
    all_answers = list(set(all_answers))
    all_answers = sorted(answer_counts.keys(), key=lambda x: answer_counts[x], reverse=True)

    print("Gathered the following answers: ", all_answers)

    # Credentials
    username = os.environ.get("EULER_USERNAME")
    password = os.environ.get("EULER_PASSWORD")
    if not password or not username:
        raise ValueError(
            "EULER_PASSWORD or EULER_USERNAME environment variable not set. Please set it to your Project Euler password/username."
        )

    if not os.path.exists(f"logs/euler/{problem_id}"):
        os.makedirs(f"logs/euler/{problem_id}", exist_ok=True)
    submitted_answers = []
    for answer in all_answers:
        html_path = f"logs/euler/{problem_id}/submission_{answer}.html"
        if os.path.exists(html_path):
            print(f"Answer {answer} already submitted, skipping")
            submitted_answers.append(answer)
    all_answers = [answer for answer in all_answers if answer not in submitted_answers]
    print("Submitting the following answers: ", all_answers)

    # Launch Playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        print("before login")

        # Try to login with cookies first (unless force-login is specified)
        login_successful = False
        if not args.force_login:
            login_successful = login_with_cookies(page, context)

        # If cookie login failed or force-login was specified, do manual login
        if not login_successful:
            login_successful = manual_login(page, username, password)
            if login_successful:
                # Save cookies after successful manual login
                save_cookies(context)

        if not login_successful:
            print("Login failed. Exiting.")
            browser.close()
            return

        print("Login successful. Starting submissions...")

        # 2. Submit each answer
        for answer in all_answers:
            is_submitted = False
            while not is_submitted:
                print("-> Submitting answer: ", answer)
                page.goto(PROBLEM_URL)
                # Check if we're already finished (there is div with id problem_answer)
                problem_answer_element = page.query_selector("#problem_answer")
                if problem_answer_element and "Completed on" in problem_answer_element.inner_text():
                    print("Problem is already solved, stopping!")
                    exit(0)
                # Check for captcha and handle it
                captcha_element = page.query_selector("#captcha")
                if captcha_element:
                    solve_captcha(page)
                try:
                    page.fill(f'input[name="guess_{euler_id}"]', str(answer))
                    page.click('input[value="Check"]')
                    print(f"Submitted answer: {answer}")
                except Exception as e:
                    print(f"Error submitting {answer}: {e}")

                # Wait for the result to load
                page.wait_for_load_state("networkidle")

                # Get the page content
                result_content = page.content()

                if "The confirmation code you entered was not valid" in result_content:
                    print("Your captcha was not correct, try again (no delay)")
                else:
                    # Take screenshot of the submission result
                    html_path = f"logs/euler/{problem_id}/submission_{answer}.html"
                    with open(html_path, "w") as f:
                        f.write(page.content())
                    if 'src="images/clipart/answer_correct.png"' in result_content:
                        print(f"Answer {answer} is correct!")
                        # store it in answers.csv
                        path = f"data/euler/{euler}/answers.csv"
                        csv_res = pd.read_csv(path)
                        # get the row with the problem_id
                        if problem_id in csv_res["id"].values:
                            csv_res.loc[csv_res["id"] == problem_id, "answer"] = answer

                        csv_res.to_csv(path, index=False)
                        # reparse all answers by running python scripts/reparse_all.py --comp euler/euler
                        subprocess.run(["python", "scripts/curation/reparse_all.py", "--comp", f"euler/{euler}"])
                        exit(0)

                    print(f"Saved submission result HTML to {html_path}")
                    is_submitted = True
                    screenshot_path = f"logs/euler/{problem_id}/submission_{answer}.png"
                    page.screenshot(path=screenshot_path)
                    print(f"Saved submission result screenshot to {screenshot_path}")

                    # Random delay
                    delay = random.uniform(MIN_DELAY, MAX_DELAY)
                    print(f"Waiting {delay:.2f} seconds before next submission...")
                    time.sleep(delay)

        print("All submissions complete.")
        browser.close()


if __name__ == "__main__":
    main()
