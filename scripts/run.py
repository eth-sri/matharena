import argparse
import json
from collections import defaultdict
from dotenv import load_dotenv
from loguru import logger


from matharena.request_logger import request_logger
from matharena.runner import Runner
from matharena.solvers import PureModelSolver, SolverResponse

def _client_signature(client_args):
    # Share one run_queries call only for compatible APIClient configs.
    normalized = {}
    for key, value in client_args.items():
        if key == "tools":
            tools = []
            for func, spec in value:
                func_name = None if func is None else getattr(func, "__name__", str(func))
                tools.append([func_name, spec])
            normalized[key] = tools
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True, default=str)


def _run_combined_pure_model_group(model_name, group):
    combined_queries = []
    combined_meta = []
    global_batch_idx_to_problem_idx = {}
    shared_client = group[0][1]["solver"].client

    for runner, prepared in group:
        solver = prepared["solver"]
        for local_idx, stmt in enumerate(prepared["batch"]):
            combined_queries.append(solver.build_query(stmt[0], stmt[1]))
            combined_meta.append((runner, prepared, local_idx))
            global_batch_idx_to_problem_idx[len(combined_meta) - 1] = prepared["batch_idx_to_problem_idx"][local_idx]

    if len(group) > 1:
        request_logger.set_metadata("multi", model_name, global_batch_idx_to_problem_idx)
    else:
        request_logger.set_metadata(group[0][0].comp_name, model_name, global_batch_idx_to_problem_idx)
    logger.info(
        f"Running a shared run_queries call for model {model_name}: {len(combined_queries)} queries across {len(group)} competitions."
    )

    for global_idx, conversation, detailed_cost in shared_client.run_queries(combined_queries):
        runner, prepared, local_idx = combined_meta[global_idx]
        solver_response = SolverResponse(local_idx, conversation, detailed_cost, history=None)
        runner.process_solver_responses(
            solver_name=prepared["solver_name"],
            solver=prepared["solver"],
            all_runs=prepared["all_runs"],
            batch_idx_to_problem_idx=prepared["batch_idx_to_problem_idx"],
            status_path=prepared["status_path"],
            solver_responses=[solver_response],
            print_final_status=False,
        )
    for runner, prepared in group:
        runner.print_final_status(prepared["status_path"])

if __name__ == "__main__":
    load_dotenv()

    # Main args: which competition to run with which models; how many runs per problem
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", type=str, nargs="+", required=True, help="Competition config(s) to run")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of model configs to run, might have scaffolding, example: xai/grok-4",
    )
    parser.add_argument("--n", type=int, default=4, help="Number of runs per problem")
    parser.add_argument(
        "--comp-n",
        type=str,
        nargs="*",
        default=[],
        help="Per-comp run count overrides in the form comp=n (e.g. apex/apex_2025=16)",
    )
    parser.add_argument(
        "--problems",
        type=int,
        nargs="+",
        required=False,
        help="List of 1-based problem indices to run, example: 1 2 3 (default: all problems in competition)",
    )

    # skip-existing is default
    parser.add_argument(
        "--redo-all", action="store_true", help="Redo all (model, problem) pairs regardless of existing runs"
    )

    # Generally ok to keep defaults here
    parser.add_argument("--comp-configs-dir", type=str, default="configs/competitions")
    parser.add_argument("--model-configs-dir", type=str, default="configs/models")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    comp_n_overrides = {}
    for override in args.comp_n:
        if "=" not in override:
            raise ValueError(f"Invalid --comp-n entry '{override}'. Expected format comp=n.")
        comp_name, n_value = override.split("=", 1)
        if comp_name not in args.comp:
            raise ValueError(f"--comp-n override '{override}' references comp not in --comp list.")
        try:
            comp_n_overrides[comp_name] = int(n_value)
        except ValueError as exc:
            raise ValueError(f"Invalid n in --comp-n override '{override}'.") from exc

    logger.info(f"Initializing runners for competitions: {args.comp}")
    runners = [
        Runner(
            comp,
            comp_n_overrides.get(comp, args.n),
            args.problems,
            args.comp_configs_dir,
            args.model_configs_dir,
            args.output_dir,
            args.redo_all,
        )
        for comp in args.comp
    ]

    # Run each model
    for model in args.models:
        logger.info(f"Calling runner for model: {model}")
        prepared_runs = []
        for runner in runners:
            prepared = runner.prepare_run(model, set_request_metadata=False)
            if prepared is not None:
                prepared_runs.append((runner, prepared))

        if len(prepared_runs) == 0:
            logger.info(f"No pending runs for model {model} in the selected competitions.")
            continue

        grouped = defaultdict(list)
        for runner, prepared in prepared_runs:
            solver = prepared["solver"]
            if not isinstance(solver, PureModelSolver):
                grouped[("agent", id(runner))].append((runner, prepared))
                continue
            signature = _client_signature(solver.default_api_client_args)
            grouped[("pure_model", signature)].append((runner, prepared))

        for group_key, group in grouped.items():
            group_type = group_key[0]
            if group_type == "pure_model":
                _run_combined_pure_model_group(model, group)
            else:
                # Agent solvers cannot be merged into one run_queries call.
                runner, prepared = group[0]
                responses = prepared["solver"].solve_batch(
                    prepared["batch"], prepared["batch_idx_to_problem_idx"], prepared["batch_idx_to_run_idx"]
                )
                runner.process_solver_responses(
                    solver_name=prepared["solver_name"],
                    solver=prepared["solver"],
                    all_runs=prepared["all_runs"],
                    batch_idx_to_problem_idx=prepared["batch_idx_to_problem_idx"],
                    status_path=prepared["status_path"],
                    solver_responses=responses,
                )