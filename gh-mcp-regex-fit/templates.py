#!/usr/bin/env python3
"""Two disjoint query families over the 79 routing targets, plus off-topic rows.

Family **A** trains the fitter. Family **B** is the held-out surface: written to
name the same intent with different verbs and a different sentence shape, and
filled from a disjoint entity pool. The point of the split is the one thing
`monad-bsky` could check about its hand-written rules — do they survive a
phrasing they were never fitted on.

Both families share an author, which is a weaker guarantee than two authors
would be. `wild.jsonl` exists for that reason: it was written and committed
before the fitter was run.

Slots: {or} owner/repo, {pr} PR ref, {issue} issue ref, {sha}, {path},
{branch}, {base}, {tag}, {run} run id, {job} job id, {wf} workflow file,
{q} free text, {user}, {org}, {team}, {thread}, {title}.
"""

# label: (family A templates, family B templates)
T: dict[str, tuple[list[str], list[str]]] = {
 "actions_get::get_workflow": (
  ["show the {wf} workflow definition in {or}", "get workflow {wf} for {or}"],
  ["what does {or}'s {wf} pipeline config actually say?", "pull up the {wf} workflow spec over on {or}"]),
 "actions_get::get_workflow_run": (
  ["get workflow run {run} in {or}", "show run {run} for {or}"],
  ["how did run {run} end up on {or}?", "give me the summary of {or} run {run}"]),
 "actions_get::get_workflow_job": (
  ["get workflow job {job} in {or}", "show job {job} details for {or}"],
  ["what went on inside job {job} over at {or}?", "walk me through {or} job {job}"]),
 "actions_get::download_workflow_run_artifact": (
  ["download artifact {job} from run {run} in {or}", "fetch the artifact {job} for {or}"],
  ["can you grab artifact {job} off {or} run {run}?", "save artifact {job} from {or} locally"]),
 "actions_get::get_workflow_run_usage": (
  ["get billable usage for run {run} in {or}", "show run {run} usage minutes for {or}"],
  ["how many minutes did {or} run {run} burn?", "what did run {run} cost {or} in CI time?"]),
 "actions_get::get_workflow_run_logs_url": (
  ["get the log download url for run {run} in {or}", "show the logs url for {or} run {run}"],
  ["where can I download the raw logs of {or} run {run}?", "link me to the full log archive for run {run} on {or}"]),
 "actions_list::list_workflows": (
  ["list workflows in {or}", "show all workflows for {or}"],
  ["which pipelines does {or} have configured?", "enumerate every workflow {or} defines"]),
 "actions_list::list_workflow_runs": (
  ["list workflow runs for {wf} in {or}", "show recent runs of {wf} in {or}"],
  ["what runs has {or}'s {wf} had lately?", "give me the run history for {wf} on {or}"]),
 "actions_list::list_workflow_jobs": (
  ["list jobs for run {run} in {or}", "show the jobs in {or} run {run}"],
  ["which jobs did run {run} on {or} contain?", "break run {run} of {or} down by job"]),
 "actions_list::list_workflow_run_artifacts": (
  ["list artifacts for run {run} in {or}", "show artifacts produced by {or} run {run}"],
  ["what files did {or} run {run} leave behind?", "enumerate the artifacts attached to run {run} on {or}"]),
 "actions_run_trigger::run_workflow": (
  ["run the {wf} workflow in {or}", "trigger {wf} on {branch} in {or}"],
  ["kick off {or}'s {wf} pipeline please", "start a fresh {wf} build for {or}"]),
 "actions_run_trigger::rerun_workflow_run": (
  ["rerun run {run} in {or}", "rerun workflow run {run} for {or}"],
  ["can you retry {or} run {run} from the top?", "run {run} on {or} again, all of it"]),
 "actions_run_trigger::rerun_failed_jobs": (
  ["rerun the failed jobs for run {run} in {or}", "retry only failed jobs in {or} run {run}"],
  ["just redo the jobs that broke in {or} run {run}", "retry the red jobs of run {run} on {or}, not the green ones"]),
 "actions_run_trigger::cancel_workflow_run": (
  ["cancel run {run} in {or}", "cancel the workflow run {run} for {or}"],
  ["stop {or} run {run}, it's wasting minutes", "abort run {run} over on {or}"]),
 "actions_run_trigger::delete_workflow_run_logs": (
  ["delete the logs for run {run} in {or}", "remove workflow run logs {run} from {or}"],
  ["purge the stored logs of {or} run {run}", "wipe run {run}'s logs on {or}"]),
 "add_comment_to_pending_review": (
  ["add a comment to my pending review on {pr} in {or}", "add a review comment on {path} to the pending review for {pr}"],
  ["stick a note on {path} into the draft review I have open for {pr}", "queue up a line comment for {pr} in my unsubmitted review"]),
 "add_issue_comment": (
  ["comment on {issue} in {or}", "add a comment to issue {issue} saying {q}"],
  ["leave a note under {issue} on {or}", "reply on the {or} issue {issue} thread"]),
 "add_reply_to_pull_request_comment": (
  ["reply to review comment {job} on {pr} in {or}", "add a reply to the comment {job} on {pr}"],
  ["respond to that reviewer's comment {job} on {or} {pr}", "answer comment {job} under {pr}"]),
 "assign_copilot_to_issue": (
  ["assign copilot to {issue} in {or}", "have copilot take issue {issue} in {or}"],
  ["put the coding agent on {or} {issue}", "hand {issue} over to copilot on {or}"]),
 "create_branch": (
  ["create branch {branch} in {or}", "make a new branch {branch} off {base} in {or}"],
  ["cut a branch called {branch} on {or}", "start {branch} from {base} over in {or}"]),
 "create_or_update_file": (
  ["create {path} in {or} on branch {branch}", "update {path} in {or} with new contents"],
  ["write a new {path} into {or}", "commit a change to {path} on {or}'s {branch}"]),
 "create_pull_request": (
  ["open a pull request from {branch} into {base} in {or}", "create a PR in {or} titled {title}"],
  ["raise a PR off {branch} against {base} on {or}", "put up a pull request for {branch} on {or}"]),
 "create_repository": (
  ["create a repository named {q}", "make a new repo called {q}"],
  ["spin up a fresh repo named {q} for me", "I want a brand new repository, call it {q}"]),
 "delete_file": (
  ["delete {path} from {or}", "remove the file {path} in {or} on {branch}"],
  ["get rid of {path} over on {or}", "drop {path} out of {or}"]),
 "fork_repository": (
  ["fork {or}", "fork the repo {or} to my account"],
  ["make me a fork of {or}", "copy {or} into my own namespace"]),
 "get_commit": (
  ["get commit {sha} in {or}", "show commit {sha} details for {or}"],
  ["what did {sha} change on {or}?", "open up {or}@{sha} for me"]),
 "get_file_contents": (
  ["get the contents of {path} in {or}", "show {path} from {or} on branch {branch}"],
  ["read me {path} out of {or}", "what's currently in {or}'s {path}?"]),
 "get_job_logs": (
  ["get the logs for job {job} in {or}", "show failed job logs for run {run} in {or}"],
  ["why did job {job} fail on {or}? show the log", "print the log output of {or} job {job}"]),
 "get_label": (
  ["get the {q} label in {or}", "show details of label {q} for {or}"],
  ["what colour and description does {or}'s {q} label have?", "look up the label named {q} on {or}"]),
 "get_latest_release": (
  ["get the latest release of {or}", "show {or}'s most recent release"],
  ["what version did {or} ship last?", "what's the newest release out of {or}?"]),
 "get_me": (
  ["get my github profile", "who am I on github"],
  ["what account am I authenticated as?", "show my own user details"]),
 "get_release_by_tag": (
  ["get release {tag} of {or}", "show the {tag} release for {or}"],
  ["what shipped in {or}'s {tag}?", "pull up release notes for {tag} on {or}"]),
 "get_tag": (
  ["get tag {tag} in {or}", "show tag details for {tag} in {or}"],
  ["what commit does {or}'s {tag} tag point at?", "resolve the {tag} tag on {or}"]),
 "get_team_members": (
  ["list members of the {team} team in {org}", "get team members for {team} in {org}"],
  ["who is on {org}'s {team} team?", "show me everyone in the {team} team at {org}"]),
 "get_teams": (
  ["list my teams", "get the teams I belong to"],
  ["which teams am I a member of?", "show every team I'm in"]),
 "issue_read::get": (
  ["get issue {issue} in {or}", "show issue {issue} details for {or}"],
  ["what does {or} {issue} actually say?", "open up issue {issue} on {or}"]),
 "issue_read::get_comments": (
  ["get the comments on issue {issue} in {or}", "show issue {issue} comments for {or}"],
  ["what have people said under {or} issue {issue}?", "read me the discussion on issue {issue} of {or}"]),
 "issue_read::get_sub_issues": (
  ["list sub-issues of issue {issue} in {or}", "get the sub issues under {issue} in {or}"],
  ["what's nested under {or} issue {issue}?", "show the child issues of {issue} on {or}"]),
 "issue_read::get_parent": (
  ["get the parent issue of {issue} in {or}", "show which issue {issue} belongs to in {or}"],
  ["what issue is {or} {issue} filed under?", "find the umbrella issue above {issue} on {or}"]),
 "issue_read::get_labels": (
  ["get the labels on issue {issue} in {or}", "show which labels issue {issue} has in {or}"],
  ["how is {or} issue {issue} tagged?", "what labels are stuck on {issue} over at {or}?"]),
 "issue_write::create": (
  ["create an issue in {or} titled {title}", "open a new issue on {or} about {q}"],
  ["file a bug against {or} for {q}", "raise a ticket on {or}: {title}"]),
 "issue_write::update": (
  ["update issue {issue} in {or} to closed", "edit issue {issue} in {or} and change the title"],
  ["close out {or} issue {issue}", "retitle issue {issue} on {or}"]),
 "list_branches": (
  ["list branches in {or}", "show all branches for {or}"],
  ["what branches exist on {or}?", "enumerate {or}'s branches"]),
 "list_commits": (
  ["list commits in {or} on {branch}", "show recent commits for {or}"],
  ["what's landed on {or} lately?", "give me the commit history of {or}'s {branch}"]),
 "list_issue_fields": (
  ["list the issue fields available in {org}", "show issue fields configured for {or}"],
  ["what custom issue fields does {org} define?", "which issue fields can I set on {or}?"]),
 "list_issue_types": (
  ["list issue types for {org}", "show the issue types available in {org}"],
  ["what issue types has {org} set up?", "which kinds of issue can I file in {org}?"]),
 "list_issues": (
  ["list open issues in {or}", "show issues for {or} labelled {q}"],
  ["what's still open on {or}?", "run through {or}'s issue list"]),
 "list_pull_requests": (
  ["list open pull requests in {or}", "show PRs for {or}"],
  ["what pull requests are waiting on {or}?", "run down the open PRs over at {or}"]),
 "list_releases": (
  ["list releases for {or}", "show all releases of {or}"],
  ["what versions has {or} published?", "give me {or}'s release history"]),
 "list_repository_collaborators": (
  ["list collaborators on {or}", "show who has access to {or}"],
  ["who can push to {or}?", "which people are collaborators on {or}?"]),
 "list_tags": (
  ["list tags in {or}", "show all tags for {or}"],
  ["what tags does {or} carry?", "enumerate every tag on {or}"]),
 "merge_pull_request": (
  ["merge {pr} in {or}", "squash merge pull request {pr} in {or}"],
  ["land {pr} on {or} please", "get {pr} merged over at {or}"]),
 "pull_request_read::get": (
  ["get pull request {pr} in {or}", "show PR {pr} details for {or}"],
  ["what is {or} {pr} about?", "open up pull request {pr} on {or}"]),
 "pull_request_read::get_diff": (
  ["get the diff for {pr} in {or}", "show the diff of pull request {pr}"],
  ["what code does {or} {pr} actually change?", "show me the patch on {pr}"]),
 "pull_request_read::get_status": (
  ["get the status of {pr} in {or}", "show CI status for pull request {pr}"],
  ["is {or} {pr} green?", "did the checks pass on {pr}?"]),
 "pull_request_read::get_files": (
  ["list the files changed in {pr} in {or}", "show which files {pr} touches"],
  ["which paths does {or} {pr} modify?", "what files are in the {pr} changeset?"]),
 "pull_request_read::get_commits": (
  ["list commits on {pr} in {or}", "show the commits in pull request {pr}"],
  ["what commits make up {or} {pr}?", "break {pr} down into its commits"]),
 "pull_request_read::get_review_comments": (
  ["get the review comments on {pr} in {or}", "show inline review comments for {pr}"],
  ["what did reviewers flag inline on {or} {pr}?", "show the line-level feedback left on {pr}"]),
 "pull_request_read::get_reviews": (
  ["get the reviews on {pr} in {or}", "show who reviewed pull request {pr}"],
  ["has anyone approved {or} {pr} yet?", "what review verdicts does {pr} have?"]),
 "pull_request_read::get_comments": (
  ["get the discussion comments on {pr} in {or}", "show the conversation on pull request {pr}"],
  ["what's been said in the {or} {pr} thread?", "read me the top-level chatter on {pr}"]),
 "pull_request_read::get_check_runs": (
  ["get the check runs for {pr} in {or}", "show which checks ran on pull request {pr}"],
  ["which CI checks are attached to {or} {pr}?", "list the check runs reporting on {pr}"]),
 "pull_request_review_write::create": (
  ["create a pending review on {pr} in {or}", "start a review for pull request {pr}"],
  ["open a draft review against {or} {pr}", "begin reviewing {pr}, don't submit yet"]),
 "pull_request_review_write::submit_pending": (
  ["submit my pending review on {pr} in {or}", "submit the review for {pr} approving it"],
  ["send off the draft review I have on {or} {pr}", "publish my in-progress review of {pr}"]),
 "pull_request_review_write::delete_pending": (
  ["delete my pending review on {pr} in {or}", "discard the pending review for {pr}"],
  ["throw away the draft review sitting on {or} {pr}", "scrap my unsubmitted review of {pr}"]),
 "pull_request_review_write::resolve_thread": (
  ["resolve review thread {thread} on {pr} in {or}", "mark thread {thread} resolved on {pr}"],
  ["close out the review conversation {thread} on {or} {pr}", "that thread {thread} on {pr} is handled, resolve it"]),
 "pull_request_review_write::unresolve_thread": (
  ["unresolve review thread {thread} on {pr} in {or}", "reopen thread {thread} on {pr}"],
  ["put the review conversation {thread} on {or} {pr} back to open", "unmark {thread} as resolved on {pr}"]),
 "push_files": (
  ["push {path} and {q} to {branch} in {or} in one commit", "push multiple files to {or} on {branch}"],
  ["commit several files at once onto {or}'s {branch}", "batch these file changes into one commit on {or}"]),
 "request_copilot_review": (
  ["request a copilot review on {pr} in {or}", "have copilot review pull request {pr}"],
  ["get the AI reviewer to look at {or} {pr}", "ask copilot for a review pass on {pr}"]),
 "resolve_review_thread": (
  ["resolve thread {thread}", "mark review thread {thread} as resolved"],
  ["that conversation {thread} is done, close it", "settle review thread {thread}"]),
 "search_code": (
  ["search code for {q} in {or}", "find code matching {q} across github"],
  ["where in the source does {q} appear?", "hunt down {q} in the codebase"]),
 "search_commits": (
  ["search commits for {q} in {or}", "find commits mentioning {q}"],
  ["which commit messages talk about {q}?", "dig through commit history for {q}"]),
 "search_issues": (
  ["search issues for {q} in {or}", "find issues about {q}"],
  ["are there any tickets discussing {q}?", "look for issues that mention {q}"]),
 "search_pull_requests": (
  ["search pull requests for {q} in {or}", "find PRs about {q}"],
  ["which pull requests touch on {q}?", "look for open PRs mentioning {q}"]),
 "search_repositories": (
  ["search repositories for {q}", "find repos matching {q}"],
  ["which projects out there do {q}?", "look for repositories about {q}"]),
 "search_users": (
  ["search users for {q}", "find github users named {q}"],
  ["which accounts match {q}?", "look up people called {q} on github"]),
 "sub_issue_write": (
  ["add issue {issue} as a sub-issue of {job} in {or}", "link {issue} under parent issue {job} in {or}"],
  ["nest {or} issue {issue} beneath {job}", "make {issue} a child of issue {job} on {or}"]),
 "unresolve_review_thread": (
  ["unresolve thread {thread}", "reopen review thread {thread}"],
  ["that conversation {thread} isn't settled, reopen it", "undo the resolve on review thread {thread}"]),
 "update_pull_request": (
  ["update {pr} in {or} to change the title", "edit pull request {pr} and mark it ready for review"],
  ["retitle {or} {pr}", "take {pr} out of draft over on {or}"]),
 "update_pull_request_branch": (
  ["update the branch of {pr} in {or}", "merge the base branch into {pr}"],
  ["bring {or} {pr} up to date with its base", "sync {pr} with main so it stops conflicting"]),
}

# Off-topic: nothing in this catalogue serves them. The point of measuring these
# is that `monad-bsky`'s regex scored 0.500 refusal fitted and 0.183 unseen,
# because a catch-all fallback swallowed everything.
OFF_TOPIC = (
 ["rename this file to draft2", "what's the weather in oslo tomorrow",
  "write me a python script that reverses a string", "deploy the app to production",
  "what does the acronym CRDT stand for", "summarise this paragraph for me",
  "convert 4pm CET to eastern", "run the test suite locally",
  "install pytorch with pip", "explain how a bloom filter works",
  "add a column to the spreadsheet", "book a meeting with the team",
  "what is my current git branch", "translate this readme into norwegian",
  "generate a uuid", "sort these numbers descending",
  "who won the world cup in 1998", "refactor this function to use a dict",
  "what's the difference between a mutex and a semaphore",
  "set an alarm for 6am", "count the words in this document",
  "compress these images to webp", "what port does postgres listen on by default",
  "recommend a book about distributed systems", "fix the indentation in this yaml",
  "how much is 340 euros in dollars", "start a jupyter notebook",
  "what's the capital of estonia", "make this regex case insensitive",
  "print the last 50 lines of the log file", "define idempotent",
  "draft a standup update for today", "convert this csv to json",
  "how do I exit vim", "what's my ip address", "kill the process on port 8080",
  "spell out the phonetic alphabet", "round these floats to 2 decimals",
  "what year was the transistor invented", "brew install ripgrep"],
 ["can you tidy up my downloads folder", "how long is the flight to reykjavik",
  "draft an email to the vendor about the invoice", "restart the staging server",
  "which database should I use for time series", "make this sentence shorter",
  "what timezone is UTC+2 in july", "clear the npm cache",
  "upgrade numpy to the latest version", "why is quicksort n log n on average",
  "chart this data as a bar graph", "put a reminder in my calendar for friday",
  "show me disk usage on this machine", "spellcheck the changelog",
  "pick a random number between 1 and 100", "reverse the order of these lines",
  "who directed blade runner", "rewrite this loop as a comprehension",
  "when should I use a trie over a hash map",
  "wake me up in twenty minutes", "how many characters are in this string",
  "resize this png to 800 wide", "what's the default ssh port",
  "suggest a podcast about linguistics", "reformat this json with two-space indent",
  "convert 12 stone to kilograms", "launch the local dev server",
  "what's the population of malta", "make this glob match dotfiles",
  "tail the error log", "what does eventually consistent mean",
  "write a short bio for my conference talk", "turn this markdown table into csv",
  "how do I undo the last shell command", "what is the current unix timestamp",
  "free up the port the server is holding", "read out the nato alphabet",
  "truncate these strings to 40 chars", "what decade was tcp/ip standardised",
  "apt install jq"],
)
