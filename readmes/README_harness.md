# Harness-backed model runs

MathArena can run a pure-model configuration through a coding-agent harness from the
vendored `harness_wrapper` package. Harness use is gated by the competition so a model
configuration cannot silently turn an ordinary evaluation into an agentic one.

## Selection rules

- Competition configs default to `allow_harness: false` when the key is absent.
- With `allow_harness: true`, pure-model configs default to `harness: codex`.
- Set competition `default_harness: false` to require an explicit model `harness`
  setting instead. A harness name can select a different competition default.
- A model can select `harness: codex`, `harness: claude`, `harness: kimi`,
  `harness: gravity`, `harness: qwen`, `harness: opencode`, `harness: deepcode`, `harness: grok`,
  `harness: muse`, or `harness: vibe`.
- A model can set `harness: false` to opt out of a competition's default harness.
- With `allow_harness: false` (or absent), a requested harness is ignored with a warning.
  Every model uses its normal API path, including models requesting Codex. CLI version
  and harness settings are removed from the API arguments.
- Existing MathArena scaffold agents retain their solver paths. Legacy `type: codex_cli`
  requires `allow_harness: true`.
- `harness_config.tools_enabled: false` can also disable Codex tools in an enabled
  competition. It does not enable a harness in a competition with `allow_harness: false`.

`arxivlean/june` enables explicit harness models with `allow_harness: true` and
`default_harness: false`: existing API models keep their native Lean tools. It uses the
shared Lean 4.31 image and read-only cache. The helper scripts also use
`arxivlean/june` directly. The dedicated `arxiv/june_harness` config defaults to Codex;
`arxiv/june`, `arxiv_false/june`, and `arxivlean/march` remain unchanged.

Explicit empty completions are saved without a final-answer reprompt. This includes
native Codex `turn.completed` events with no answer. Claude and Muse also emit empty
results after exhausting their output-limit continuations, preserving reported usage
and partial work. Transient connection failures remain errors. Lean runs still check
and submit `Solution.lean` even when chat output is empty.

## ArxivMath August: SageMath and answer judging

`arxiv/august` enables harnesses and uses the shared image with SageMath 9.5.
Model configs still select their harness; the competition default is Codex.
The prompt lists the available local tools, explains `sage -c` and `sage -python`,
and specifies `/work` as the writable directory with no public internet access.
The regular scientific-Python virtualenv is unchanged; Sage uses Debian's Python.
The image pins [Debian's SageMath package](https://packages.debian.org/bookworm/sagemath).

Build the shared image when needed:

```sh
docker build -f docker/harness-arxivlean/Dockerfile -t matharena-harness-arxivlean:lean4.31 docker/harness-arxivlean
```

Generation and judging remain separate, using the existing judge framework:

```sh
uv run python scripts/run.py --comp arxiv/august --models openai/gpt-6-astra --n 1
uv run python scripts/judge/judge.py --comp arxiv/august --models openai/gpt-6-astra
```

`grading: answer_judge` bypasses parser grading and parser-triggered last-chance
requests. Every response is saved as `TODO Grading`, retaining its ground truth.
The second command uses `judges/answer_judge`: Gemini 3.8 Flash through the normal
API client (`GOOGLE_API_KEY`), with no harness or tools, compares the final response
to `answers.csv`. The ground truth is never added to the solver prompt. It records
a binary score, assessment, full judge transcript, and separate judge costs in
`judgment`. Invalid verdicts remain pending; rerunning skips successful judgments.
Use `--redo` to deliberately replace judgments. `scripts/regrade.py` will not
overwrite these scores with the parser.

## Gemini 3.8 Flash across harnesses

These configs all use `GOOGLE_API_KEY`, high reasoning, the same token prices,
65,536 maximum output tokens, and 16 concurrent requests:

| Harness | Model config | CLI version | Gemini endpoint |
| --- | --- | --- | --- |
| Antigravity CLI | `gemini/gemini-38-flash` | 1.1.26 | Native Google |
| Qwen Code | `gemini/gemini-38-flash-qwen` | 0.23.0 | Native Google |
| OpenCode | `gemini/gemini-38-flash-opencode` | 1.18.27 | Native Google |
| Kimi Code | `gemini/gemini-38-flash-kimi` | 0.40.1 | Google OpenAI-compatible Chat Completions |

Use any of these config IDs with the normal `scripts/run.py --models` option.
The existing low-reasoning configs remain available. Each harness variant has
its own output directory and display name for comparisons. Kimi's config sets
its context window to 1,048,576 tokens and enables file polling to avoid the
shared host's filesystem-watcher limit.

The three added variants passed live Gemini tool-call/response checks with high
reasoning. Deep Code 0.3.1 loses the required thought signature when it returns
a tool result; Google rejects that continuation with HTTP 400, so no Gemini
config is supplied for it. Claude Code requires an Anthropic-compatible endpoint,
while Codex and Muse Code use Responses; these routes are not implemented for
Gemini here. Supporting them requires more than a model config. Google's
[OpenAI compatibility documentation](https://ai.google.dev/gemini-api/docs/openai)
describes the Chat Completions endpoint used by Kimi.

## Grok 4.7 through OpenRouter and Grok Build

`xai/grok-47` uses `x-ai/grok-4.7` with `OPENROUTER_API_KEY`, xAI-only routing,
`xhigh` reasoning, and a 450,000-token output ceiling. For Grok Build this ceiling
is adaptive: on an explicit OpenRouter context-overflow HTTP 400, the proxy uses
the reported text/tool input counts to reserve 8,192 tokens of headroom and retries
the same request once with a smaller output allowance. That allowance is retained
for later calls and can shrink again as history grows. The input, encrypted reasoning,
tools, and native compaction remain intact; unrelated errors keep native retry handling.
A prompt that itself fills the context window is not fixed by reducing output.
Temperature and top-p are
omitted so the provider supplies its defaults. Both the normal API path and
Grok Build use OpenRouter's Responses endpoint with `store: false`. The config
pins Grok Build 1.0.40
and a 500,000-token context window for harness-enabled competitions.

May and June use the normal API path and do not install or launch a harness:

```sh
uv run python scripts/run.py --comp arxiv/may --models xai/grok-47
uv run python scripts/run.py --comp arxiv/june --models xai/grok-47
```

August and `arxivlean/june` select the native Grok Build harness automatically:

```sh
uv run python scripts/run.py --comp arxiv/august --models xai/grok-47 --n 1
```

The host downloads the versioned official CLI and verifies its reported version.
No shared-image rebuild is required. Headless sessions use a private Grok home,
file-based prompts, streaming JSON, and native `--resume`. Grok sends full history
on tool continuations and resume, as required by OpenRouter's stateless Responses
API. Native Grok inference retries are preserved, including retries of
reasoning-only responses. The existing Docker
boundary provides filesystem and network isolation; Grok's nested sandbox is off.
The real API key remains in the host proxy. Minimal mode permits local terminal,
file, search, and todo tools; the proxy enforces this even when Grok's own tool
filter retains extra tools. Provider usage includes auxiliary title requests,
reasoning, and cached input, without double-counting native usage events.

The configured reporting prices are OpenRouter's standard base rates, verified
2026-09-21. OpenRouter doubles these rates at 200,000 prompt tokens; MathArena's
static cost estimates do not currently represent that tier.

See [Grok Build's headless reference](https://docs.x.ai/build/cli/headless-scripting)
and [custom-model settings](https://docs.x.ai/build/settings/reference).

## Mistral Large 4 through Mistral Vibe

`mistral/mistral_large_4` runs `mistralai/mistral-large-4-0` through
[Mistral Vibe](https://github.com/mistralai/mistral-vibe), Mistral's own coding-agent CLI,
pinned to 2.26.1 (`harness: vibe`; aliases `vibe-cli`, `mistral-vibe`, `mistral`).
It uses `OPENROUTER_API_KEY` with Mistral-only routing, the same endpoint and pricing as the
API path of the same config. Large 4 offers only `none` and `high` reasoning;
`high` is its maximum. Vibe always sends a temperature (0.2 unless configured), so the
proxy drops it and Mistral's model default (1.0) applies, matching the API config. Set
`temperature` in a model config to send one explicitly. `max_tokens: 262144` is OpenRouter's
output ceiling for this endpoint and is added by the proxy, since Vibe sends none.

August uses the harness automatically; May and June use the normal API path:

```sh
uv run python scripts/run.py --comp arxiv/august --models mistral/mistral_large_4 --n 1
uv run python scripts/judge/judge.py --comp arxiv/august --models mistral/mistral_large_4
uv run python scripts/run.py --comp arxiv_false/august --models mistral/mistral_large_4 --n 1
uv run python scripts/judge/judge.py --comp arxiv_false/august --models mistral/mistral_large_4
```

The managed installer downloads Mistral's standalone PyInstaller build
(`vibe-linux-x86_64-<version>.zip`) from the GitHub release, verifies the release's SHA-256
digest, checks `vibe --version`, and mounts it read-only; no image rebuild is required. The
bundle carries its own Python 3.12 and the native Unified Harness runtime and needs glibc 2.28.

Sessions use `vibe -p` with the prompt on stdin, `--output streaming`, `--auto-approve`,
`--trust`, a private `VIBE_HOME`, and `VIBE_CLI=python` (the Rust CLI rejects `--resume`
together with `-p`). Retries and the final-answer reprompt use `--resume <session id>`;
the replayed history is deduplicated. Telemetry, experiments, update checks, connectors,
subagents, project context, and session titles are off, so a turn makes only model requests.
Background processes stay enabled because `bash` calls time out after 300 seconds.
Vibe's `bash` tool starts a login shell, whose `/etc/profile` replaces the sandbox `PATH`, and
inherits the PyInstaller bundle's `LD_LIBRARY_PATH`, which breaks SciPy and Sage
(`GLIBCXX_3.4.26 not found`). A private `~/.bash_profile` restores both, so commands use the
image's Python venv, Sage, Lean, and Node like the other harnesses.

Minimal mode replaces Vibe's system prompt and limits the offered tools to `bash`,
`read_file`, `write_file`, `edit`, and `todo`. Vibe's Unified Harness always adds
`search_tool_functions`, `run_typescript` (an embedded TypeScript runtime that can start
subagents and processes), `cron`, and a scratchpad; the proxy removes them and rejects
non-Chat-Completions endpoints. Filtering errors fail closed.

Vibe's programmatic output contains no token usage, so usage is counted from every proxied
provider response, including cached input. Vibe replays OpenRouter's `reasoning` field on
later turns. `model_context_window: 1048576` (Mistral's maximum) sets Vibe's compaction threshold to the window
minus `max_tokens` minus 8,192 tokens, so the output allowance always fits beside history.

Vibe 2.26 treats a mid-stream `finish_reason: "error"` as an ordinary finish, which would
save a truncated reasoning trace as an empty final answer (about 3% of June API requests
ended this way). The proxy withholds such an error event and ends the stream, so Vibe sees
an incomplete stream and resends the request; Vibe retries for up to an hour after a
request first starts. Later failures exit the CLI, and the wrapper resumes the session
within `max_recovery_attempts`. Since 2.26.1, a response cut off at `max_tokens` exits 3
instead of saving the truncated text, so it is resumed the same way.

## Meta Muse Code

`meta/muse_spark_13` runs `muse-spark-1.3` through Muse Code, pinned to
`1.0.3-R2198.1`, with `max` reasoning and `max_tokens: 1000000`. The proxy forwards
that ceiling as Responses' `max_output_tokens`; it applies to each solving
response, including reasoning. If Meta rejects the combined input/output budget
with its generic invalid-parameters HTTP 400, the adapter halves the output
allowance and retries without changing the conversation or reasoning effort.
Retries are bounded by `max_recovery_attempts`; the reduced allowance is reused
for later solver requests, and each adjustment is logged and saved in the trace. Tool calls and resumed turns can use more tokens
across the whole attempt. First-event and idle stream timeouts are both eight
hours, and Muse's first-turn minimal-effort override is disabled.

Set `META_API_KEY` in `scripts/.env`, then use the normal runner:

```sh
uv run python scripts/run.py --comp arxiv_false/august --models meta/muse_spark_13 --n 1
uv run python scripts/run.py --comp arxivlean/june --models meta/muse_spark_13 --n 1
```

Muse runs in competitions with `allow_harness: true`. Other competitions use the
normal Meta Responses API with a warning, preserving the configured reasoning
and output allowance. Set `harness: false` to explicitly use the API.

The managed installer downloads the pinned native binary, verifies Meta's checksum,
and mounts it read-only into the shared container image; no image rebuild is needed.
New sessions use `muse exec --json`. Minimal context permits only local shell, file,
search, and todo tools, disabling web tools, foreign personal context, and reminder
agents. The container provides the filesystem/network boundary. The real Meta key stays in
the host proxy; Muse receives a placeholder credential.

Meta `server_error` failures inside HTTP-200 SSE responses retry the same request
up to `max_recovery_attempts` additional times (three in the supplied config),
waiting 60 seconds before each retry. Output-budget adjustment retries also wait 60 seconds.
The proxy buffers these responses so failed partial text and tool calls never reach
Muse, while retaining reported usage. HTTP failures keep Muse's native retry handling.

All harness invocation failures, including unknown errors and invalid credentials, use
`max_recovery_attempts` (default: three additional attempts). Every wrapper retry waits
60 seconds, including authentication/quota recovery and output-limit continuations,
and resumes the existing session when available. Qwen terminal API error
placeholders and missing terminal results also trigger recovery. Tool errors remain
model-visible feedback. Cancellation and competition budgets stop retries. Native
CLI/request retries have separate limits inside each invocation.

A Muse solving turn truncated by `max_output_tokens` uses that same retry budget
in the same session and workspace, using an empty input so Muse supplies its native
placeholder. Exhausting the budget produces an empty result. For every harness, an explicit
completion with no answer is saved without a final-answer reprompt. Lean runs still
check and submit `Solution.lean`; empty text submissions are incorrect. Other incomplete
attempts remain pending. Competition time and cost limits still apply, including the
configured final-answer grace period.

Continuations use `muse serve` with `session/resume` and `turn/start`, avoiding the
`exec --session-id` journal-sequence failure after compaction. The adapter fetches
catalog metadata once on first resume and exposes only the configured model. It
preserves native context and files, and takes final text from the complete provider
response after native completion because the session protocol can truncate view items.

Compaction retains Muse's internal `generate_summary` tool, schema, prompt, reasoning,
and output allowance. Solver overrides do not replace those auxiliary settings.
Reported usage from solving, compaction, and failed requests is included in costs;
reasoning tokens are already part of output usage. Requests, tool calls/results,
and final answers are retained in the normal harness traces.

See the [adapter details](../harness_wrapper/README.md#muse-code),
[Meta's Muse Code documentation](https://meta-models.github.io/muse-code-sdk/), and
[official installer](https://dev.meta.ai/install.sh).

## Tool-free Codex and GPT-6 Astra

`openai/gpt-6-astra` uses the saved Codex subscription with max reasoning and pins CLI
0.153.3 when the competition enables harnesses. Competitions such as `arxiv/june`
and `arxiv_false/june` that disable harnesses use the normal OpenAI API instead,
with its API credentials and billing.

Tool-free Codex remains available for curation queries and for harness-enabled
competitions with `harness_config.tools_enabled: false`.

The tool-free adapter disables shell/code execution, browsing, images, subagents,
plugins, skills, MCP, and project instruction discovery. A deterministic per-run model
catalog sets local tool mode to `direct` and shell capability to `disabled`, preventing
Codex's built-in model metadata from forcing a code-mode host. This does not change the
upstream model ID. The optional `harness_config.model_context_window` preserves the
model's context limit (1,050,000 for Astra); without it the tool-free catalog uses Codex's
272,000-token fallback. Competition instructions and the problem remain in the prompt,
alongside a short instruction explaining that no tools are available.

Both the API and subscription proxies also enforce `tools: []`, `tool_choice: "none"`,
and `parallel_tool_calls: false` on every request, including resumed/recovery turns.
The raw request log records that final payload. Run history records `tools_enabled`,
so tool-free and tool-enabled invocations can be distinguished. Each independent run
still starts with a fresh workspace and private home; no personal skills are imported.

The same model config uses tools in `arxivlean/june` and `arxiv/june_harness`.
For the regular Lean competition:

```sh
uv run python scripts/run.py --comp arxivlean/june --models openai/gpt-6-astra --n 1
```

Lean harness runs require tools and cannot use text-only mode. The token costs in
the Astra config are API-equivalent reporting rates, not subscription charges. Model
availability on the account is checked only when an actual run starts.

Model ID, reasoning levels, and rates follow the
[official Astra model reference](https://developers.openai.com/api/docs/models/gpt-6-astra).
The local catalog uses Codex's documented
[model_catalog_json setting](https://learn.chatgpt.com/docs/config-file/config-reference).

## Container runtime

Docker is the default. Select native Linux Podman at execution time with
`--container-runtime podman`; model configurations do not select the backend:

```bash
uv run python scripts/run.py --comp arxiv/august --models openai/gpt-6-astra --container-runtime podman
uv run python scripts/judge/judge.py --comp arxiv/august --container-runtime podman
bash scripts/run_all_fa.sh openai/gpt-6-astra --container-runtime podman
```

The same runtime image, model proxy, CLI cache, and resource limits are used by
both runtimes. `container_image` and `container_extra_args` can customize the
image and launch flags; existing `docker_image` and `docker_extra_args` settings
remain supported for either runtime. The new keys take precedence when present.

For local rootless Podman, MathArena defaults to `--userns keep-id` so the
container's unprivileged UID/GID can write the host workspace. An explicit
`harness_config.userns` overrides `MATHARENA_CONTAINER_USERNS`, which overrides
that default. Rootful Podman does not add a user namespace by default. The host
must support rootless user namespaces and delegated cgroup v2 CPU/memory limits.
The existing proxy relay requires native Podman with Netavark and
`podman unshare --rootless-netns`; remote Podman clients are not supported.

Build the image and populate the Lean cache in Podman's storage before using it:

```bash
scripts/build_harness_arxivlean_image.sh --container-runtime podman
```

Docker and Podman have separate image/volume stores. Omitting the runtime option
uses Docker. The build helper also retains its `MATHARENA_CONTAINER_RUNTIME`
environment variable for compatibility; its command-line option takes precedence.

Podman uses an internal bridge with a host-side model proxy and a relay for
rootless networking. General network access remains controlled by
`direct_network`; cancellation removes the named container and its child tools.
The existing integration tests can target Podman without paid model requests:

```bash
MATHARENA_TEST_CONTAINER_RUNTIME=podman uv run pytest tests/test_harness_e2e.py -k 'container_sandbox or without_paid_credits'
```

The option is also available in judge comparison and the source-curation screening,
investigation, and verification commands. `scripts/run_all_fa.sh`,
`scripts/run_all_kangaroo.sh`, `scripts/run_euler.sh`, and the ArXivMath/BrokenArXiv
creation workflows forward it to model evaluation or judging. The training evaluation
wrapper `train/scripts/eval_arxiv_march_vllm.sh` forwards it too. Library callers can pass
`container_runtime="podman"` to `Runner`, `HarnessSolver`, `JudgePool`, or
`create_query_client`. Legacy `type: codex_cli` and the separate code-execution
tools retain their Docker integration.

## Model and authentication configuration

Codex subscription authentication is the default in the supplied Codex model config:

```yaml
harness: codex
model: gpt-5.6-sol
api: openai
reasoning_effort: max

harness_config:
  auth: subscription
  oauth_auto_login: false
  oauth_auto_relogin: false
  container_executable: codex
  direct_network: false
  read_only_rootfs: true
```

Run `codex login --device-auth` on the trusted host first. The wrapper imports the saved
native session into its mode-0600 host credential store, then exposes a Responses proxy on
an invocation-local internal container network. Neither `~/.codex/auth.json` nor the real
access or refresh token is mounted, copied, or placed in the container environment. Codex
inside the container receives only a random one-time bearer capability for that proxy.
Automatic interactive login and re-login are disabled so benchmark workers cannot stop on
an unexpected browser/device flow.

This follows OpenAI's guidance that `codex exec` can reuse saved authentication on a
trusted runner while treating `~/.codex/auth.json` like a password. Do not use subscription
credential reuse on public or untrusted runners. See
[OpenAI's non-interactive Codex documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

API-credit mode uses the same isolation:

```yaml
harness: kimi
model: glm-5.3
api: glm

harness_config:
  auth: api
  container_executable: kimi
```

MathArena resolves `GLM_API_KEY` and the Z.AI OpenAI-compatible endpoint on the host.
Sampling, reasoning, and `extra_body` values are applied by the host proxy to requests
generated by Kimi, Qwen, or OpenCode. Antigravity CLI uses Google's native Gemini endpoint;
MathArena translates its sampling and nested Google thinking configuration to the native
request shape. Every child sees only a placeholder credential. The adapter observes every
proxied Gemini response, including Antigravity's separate conversation-title request, so
those input, output, and cached tokens are included in reported usage.

Every final post-override request body seen by the harness proxy is also appended to
`<workspace>/.harness_wrapper/model_requests.jsonl`. This includes auxiliary title calls,
tool-result continuations, retries, and last-chance turns. Each JSONL record contains the
request path and exact JSON payload sent upstream; HTTP headers and API credentials are not
recorded. The relative file path is stored as `model_requests` in the run's output history.

Antigravity CLI, Qwen Code, OpenCode, and Deep Code support API-credit auth only in
MathArena's isolated adapter. Select them in an ordinary model config with
`harness: gravity`, `harness: qwen`, `harness: opencode`, or `harness: deepcode`; do not create
competition-specific copies of the model config. Antigravity requires `api: google`. Qwen
accepts native Google, OpenAI-compatible, or Anthropic-compatible endpoints; OpenCode
accepts native Google or OpenAI-compatible endpoints. With `api: google`, Qwen and
OpenCode use their native Gemini provider SDKs so thought signatures survive tool-result
continuations.

The shared image is version-neutral and contains Node.js plus the Lean/Python runtime, but
no pinned coding CLI. `harness_version` in each model config selects an exact release. If
omitted, `latest` is resolved once at run startup. MathArena installs the resolved CLI into
a versioned host cache, verifies it, mounts that package read-only at `/opt/harness-cli`,
and records both the requested and resolved version in output history. Harness selection
changes the mounted CLI and authentication adapter, not the container image.

By default the container can reach only its host proxy; `direct_network: true` should be
reserved for workloads that genuinely need public network access.

Codex's inner Bubblewrap command sandbox is disabled in the shared image because nested
user and mount namespaces are blocked by the container boundary. Model-generated commands
remain isolated by the outer container: it runs as the host's unprivileged UID/GID,
drops all capabilities, enables `no-new-privs`, uses a read-only root filesystem and an
internal-only network, and exposes only the per-problem workspace as writable.

For non-Lean competitions, Lean is merely present in the shared image: Mathlib is not mounted
and the prompt does not advertise Lean. The problem and complete instruction are passed
directly to the harness CLI rather than duplicated into prompt files. Lean competitions
initialize only `Solution.lean` and `check.sh` in addition to native per-run state.

Competition configs may set `harness_instruction` to a complete prompt used when a
harness is selected; otherwise `instruction` is used. Model `custom_instructions`
for that competition still take precedence. The solver does not append Lean-specific
instructions. `arxivlean/june.yaml` defines its file-based Lean workflow in
`harness_instruction`, while its ordinary API prompt describes the function tools.

## Deep Code / DeepSeek V4 Flash

[DeepSeek's integration guide](https://api-docs.deepseek.com/quick_start/agent_integrations/deepcode/)
links to the community-maintained [Deep Code CLI](https://github.com/lessweb/deepcode-cli).
The npm package is `@vegamo/deepcode-cli` and its executable is `deepcode`.
Use `harness: deepcode` (aliases: `deepcode-cli`, `deep-code`). The existing
`deepseek/deepseek_v4_flash` model config pins version `0.3.1`, uses `api: deepseek`
and `DEEPSEEK_API_KEY`, and retains its max reasoning and generation settings.
The shared container image already supplies the required Node.js 22 and scientific tools;
the managed installer mounts the pinned CLI from its private versioned cache.

```sh
uv run python scripts/run.py --comp arxiv/august --models deepseek/deepseek_v4_flash --n 1
uv run python scripts/judge/judge.py --comp arxiv/august --models deepseek/deepseek_v4_flash
```

Deep Code runs with native `--exec --prompt`; retries use its saved session ID with
`--resume`. Fresh problem attempts still reset their workspace. Since `--exec` emits
only final-answer text, the adapter imports native session JSONL messages after each
turn, preserving reasoning, local tool calls/results, and session identity. Provider
errors are logged by the proxy; failed tool results stay in transcripts, not log warnings.
Every post-filter request is recorded in `.harness_wrapper/model_requests.jsonl` without
credentials. Usage is taken from provider responses, including DeepSeek's
`prompt_cache_hit_tokens`; cached input is included in total input and priced separately.

Minimal mode disables telemetry, MCP, image uploads, and discovered/bundled skills.
Because the CLI has no tool allowlist flag, the proxy allows only `bash`, `read`, `write`,
`edit`, and `UpdatePlan`, replaces the stock tool prompt, and rejects non-Chat-Completions
endpoints. Filtering errors fail closed. Native permissions additionally deny network
and MCP access, and the container blocks public networking. The CLI's container-local runtime
metadata remains; no host skills or settings are mounted.

## Reproducible CLI context

MathArena passes `minimal_context: true` to the wrapper by default. This is independent of
the model configuration and can be disabled explicitly with
`harness_config.minimal_context: false`.

Minimal mode keeps native authentication and session resume, but removes optional context
at CLI invocation time:

- Codex ignores user configuration and exec-policy rules, replaces its stock instructions,
  disables bundled and discovered skills, sets the project-document budget to zero, clears
  MCP servers, and disables browser, plugin, goal, and collaboration features. It retains the
  local Code Mode host because GPT-5.6 Codex models require it to reach the shell executor;
  that runtime does not load host skills or restore internet access.
- Claude uses `--safe-mode`, `--disable-slash-commands`, an empty strict MCP configuration,
  a replacement system prompt, and `--tools Bash`.
- Kimi uses `--skills-dir` with an empty directory and a generated, deterministic
  `--agent-file` whose only enabled tool is `Bash`. The agent prompt contains no skill,
  project-instruction, plugin, or base-prompt template variables.
- Antigravity uses a fresh private home, disables slash expansion, and selects a custom agent
  whose declared tools are limited to local files, local search, and shell commands. Those
  commands run inside MathArena's hardened outer container sandbox; the CLI's nested sandbox is
  incompatible with the container's `no_new_privs` boundary. The agent declares no skills, plugins,
  MCP servers, web tools, or subagents. The closed-source
  CLI still injects its built-in messaging/artifact instructions and current-time metadata;
  these cannot currently be disabled through documented headless flags.
- Qwen uses `--safe-mode` and `--bare`, overrides its system prompt, and excludes its web
  and delegation tools.
- OpenCode uses `--pure` with an inline provider-only config. Project instructions, plugins,
  MCP, formatters, LSP, skills, subagents, and web tools are disabled.

The per-problem container home remains isolated as a second layer, so no host CLI settings,
credentials, skills, plugins, or generic `.agents` content are mounted into a run.

## arXivLean strategy

The harness uses its native file and shell tools rather than MathArena function calling:

- `verify_lean` becomes editing `Solution.lean` and running `./check.sh`.
- `add_to_file` becomes adding helper declarations directly to `Solution.lean`.
- `verify_submission` remains the normal post-run MathArena grader/comparator; `check.sh` is
  only the fast iterative compilation check.
- `loogle` and `lean_explore_search` are not automatically available inside a coding
  harness. The optimal extension is to install small same-named CLI adapters and their data
  in the shared container image. This keeps the interface harness-neutral and avoids
  provider-specific function-tool or MCP wiring.

`Solution.lean` and `check.sh` are created in each problem workspace; the full problem is
already in the CLI prompt. Native session state is stored under the isolated per-run home so
a last-chance turn can resume across disposable containers. The Mathlib cache is mounted
read-only, and the configured toolchain is checked against `lean_environment` so a Lean
4.31 image cannot silently run a Lean 4.29 benchmark.

Every independent problem/run invocation deletes and recreates its resolved workspace
before starting the CLI. Thus `keep_workspaces: true` retains artifacts for inspection
after a run, but never feeds those artifacts or old native session state into a rerun.
Only a `last_chance` continuation within the same invocation reuses that run's workspace
and native session.

The shared Lean 4.31 image supports the configured harnesses for arXivLean June. March remains
default-off until a version-matched Lean 4.29 image and cache are added.

## Running the June harness

Build the combined image and Mathlib cache once:

```bash
scripts/build_harness_arxivlean_image.sh
```

Codex uses the saved ChatGPT subscription by default; API credits are explicit:

```bash
codex login --device-auth
uv run python scripts/run.py --comp arxivlean/june --models openai/codex-max --n 1

export OPENAI_API_KEY=...
uv run python scripts/run.py --comp arxivlean/june --models openai/codex-max-api --n 1
```

Kimi with GLM always uses Z.AI API credits:

```bash
export GLM_API_KEY=...
uv run python scripts/run.py --comp arxivlean/june --models glm/glm-53-kimi --n 1
```

The container integration test uses a local fake
OpenAI-compatible server and therefore spends no credits. Paid GLM and live subscription
smokes are separately opt-in through `MATHARENA_RUN_PAID_HARNESS_E2E=1` and
`MATHARENA_RUN_SUBSCRIPTION_E2E=1`. The final arXivLean comparator remains part of the
normal grading path.


## Per-question limits

`arxiv/august` and `arxiv_false/august` configure these limits under
`harness_config` in their existing competition YAML files:

```yaml
max_time_seconds: 43200
max_cost_usd: 100
cost_limit_grace_seconds: 300
cost_limit_grace_prompt: |
  You have hit the cost budget. You must produce your final answer within the next {cost_limit_grace_minutes} minutes or you will not receive any positive score. Use your existing reasoning and saved work to give the best answer you can now.
  Your final deadline is {cost_limit_deadline_at} (UTC). You can check the current time with `date -u`. The original overall time limit still applies.
```

These apply to each coding-harness attempt, across all internal requests, tools,
retries, and continuations. Each sampled run gets its own budget. Other
competitions and direct API/scaffold solvers are unchanged. The clock starts
when the attempt starts, after waiting for a worker and preparing its workspace
and model. The configured `time_limit_prompt` is appended to the task with the
actual UTC start time and deadline; it uses `{run_started_at}`,
`{run_deadline_at}`, and `{run_time_limit_hours}` placeholders.

The dollar limit uses the model config's input, output, and cache token prices,
including reasoning tokens. It is disabled if input or output pricing is missing;
the time limit still applies. Native and provider-reported usage are reconciled
without adding the same tokens twice. Usage can arrive only at the end of a
request, so a request may cross the dollar limit before it can be stopped.

At a limit, the wrapper interrupts the CLI and forcibly removes its container
if it has not exited after two seconds. With `cost_limit_grace_seconds` and
`cost_limit_grace_prompt` configured, hitting the dollar limit then grants one
final-answer turn in the same native session and workspace. Its prompt includes
the actual UTC deadline. The grace period starts when this final turn is launched,
includes retries and tool use, and is shortened if the original 12-hour deadline
comes first. There is no extension for hitting the time limit itself.

The dollar cap is suspended during this final period, so total spending can exceed
$100; all reported usage remains counted. A nonempty final response submitted in
time is graded normally. The final chance cannot restart or earn another grace
period. Omitting the grace settings retains an immediate stop at the cost limit.

If the final deadline expires, resuming fails, no resumable session exists, or no
final answer arrives, the attempt is saved as incorrect and the judge skips it.
Partial work, the final warning prompt, and available usage are retained in the
normal result. `detailed_costs[].run_limits.cost_limit_grace` records its start,
deadline, and whether it completed; `exceeded` is null for a successful final
answer and records the failure otherwise. Interrupted usage remains flagged as
potentially incomplete. Existing outputs are not retroactively changed, and
running processes need a new launch to pick up these settings.


Set `harness_config.direct_network: true` to enable shell networking and native web tools;
it defaults to false.
