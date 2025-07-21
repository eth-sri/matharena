import asyncio
import argparse
import json
import os
import re
import regex
from openai import AsyncOpenAI
from tqdm import tqdm
JUDGE_PROMPT = """You are a math expert that has a task to check if a proposed answer to a math problem is equivalent to the gold answer.
You should check for exact equivalence of the two answers. 
You should ignore spaces, punctuation and other irrelevant characters.
For example, "x=5", "5", "5 degrees" are all considered equivalent.
Think step by step about the equivalence of the two answers.
If they are equivalent, write "EQUIVALENT" in your last sentence. If they are not equivalent, write "NOT EQUIVALENT".

# The gold answer is:
{gold_answer}

# The proposed answer is:
{answer}
"""

BOXED_PATTERN = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)

client = AsyncOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPENROUTER_API_KEY"],
)

async def judge_answer(answer, gold_answer, n_samples=4):
    responses = []
    for i in range(n_samples):
        response = await client.chat.completions.create(
            model="google/gemini-2.5-flash-preview",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(answer=answer, gold_answer=gold_answer)}],
            n=n_samples
        )
        text_response = response.choices[0].message.content
        judge_res = 0
        if "NOT EQUIVALENT" in text_response:
            judge_res = 0
        elif "EQUIVALENT" in text_response:
            judge_res = 1
        elif "not equivalent" in text_response.lower():
            judge_res = 0
        elif "equivalent" in text_response.lower():
            judge_res = 1
        else:
            judge_res = None
        if judge_res is not None:
            responses.append((judge_res, text_response))
    return responses

async def run_judgement(candidates, batch_size, n_samples=4):
    all_files = []
    for i in tqdm(range(0, len(candidates), batch_size)):
        batch = candidates[i:min(i+batch_size, len(candidates))]
        tasks = [judge_answer(answer, gold_answer, n_samples) for answer, gold_answer, correct, file_path, sample_idx in batch]
        results = await asyncio.gather(*tasks)
        for result, (answer, gold_answer, correct, file_path, sample_idx) in zip(results, batch):
            if len(result) > 0:
                judge_score = sum(res[0] for res in result) / len(result)
                if (judge_score > 0.5 and (not correct)) or (judge_score < 0.5 and correct):
                    print(f"Found potential discrepancy for {file_path}\n")
                    print(f"judge_score={judge_score}, correct={correct}\n")
                    print("Answer:\n", answer)
                    print("Gold answer:\n", gold_answer)
                    judge_path = os.path.join("parser_judge_logs", "judge_" + file_path.replace("/", "--") + f"--sample-{sample_idx}.txt")
                    all_files.append(file_path)
                    with open(judge_path, "w") as f:
                        f.write(f"judge_score={judge_score}, correct={correct}\n")
                        f.write("Answer:\n")
                        f.write(str(answer) + "\n")
                        f.write("Gold answer:\n")
                        f.write(str(gold_answer) + "\n")
                        f.write("Judge responses:\n")
                        for judge_idx, res in enumerate(result):
                            f.write(f"Judge response {judge_idx}:\n")
                            f.write(res[1])
                            f.write("\n")                    
                    print("=" * 100)

    print("Summary:")
    print(f"Total files checked: {len(all_files)}")
    print(f"Total discrepancies found: {len(all_files)}")
    print("All files with discrepancies:")
    for file in all_files:
        print(file)
    
def extract_boxed(content):
    if "\\boxed" not in content:
        return None
    pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, content))
    if matches:
        return matches[-1].group(2)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=4)
    args = parser.parse_args()

    comp_dir = os.path.join("outputs", args.comp)

    candidates = []

    for org in os.listdir(comp_dir):
        for model in os.listdir(os.path.join(comp_dir, org)):
            for file in os.listdir(os.path.join(comp_dir, org, model)):
                if file.endswith(".json"):
                    # extract the number before json
                    file_num = int(file.split("/")[-1].split(".")[0])
                    file_path = os.path.join(comp_dir, org, model, file)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        gold_answer = data["gold_answer"]
                        for i, messages in enumerate(data["messages"]):
                            if len(messages) == 2:
                                response = messages[1]["content"]
                                answer = extract_boxed(response)
                                if answer is None: # no answer found
                                    continue
                                if answer.strip().replace(" ", "").lower() == gold_answer.strip().replace(" ", "").lower() and data["correct"][i]: # it's trivially correct
                                    continue
                                # If both are integers, and they are different, and data["correct"][i] is false, it's trivially incorrect
                                try:
                                    answer_num = int(answer.strip())
                                    gold_num = int(gold_answer.strip())
                                    if answer_num != gold_num and not data["correct"][i]:
                                        continue
                                except (ValueError, TypeError):
                                    pass

                                if answer is not None:
                                    print("Adding a candidate:\n")
                                    print("Answer:\n", answer)
                                    print("Gold answer:\n", gold_answer)
                                    print("=" * 100)
                                    candidates.append((answer, gold_answer, data["correct"][i], file_path, i))

    if not os.path.exists("parser_judge_logs"):
        os.makedirs("parser_judge_logs")
    asyncio.run(run_judgement(candidates, args.batch_size, n_samples=args.n_samples))

if __name__ == "__main__":
    main()
