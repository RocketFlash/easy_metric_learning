# CLAUDE.md

See [AGENTS.md](AGENTS.md) for repo orientation, setup, conventions, and the
list of known gotchas and gaps. Everything in AGENTS.md applies to Claude Code
sessions — don't duplicate it here.

Claude-specific notes only:

- This repo is a research framework, not a product. Prefer surgical changes
  over refactors; the user iterates fast and dislikes drive-by cleanup.
- When proposing additions (new margins, losses, backbones), default to one
  concise recommendation with the tradeoff rather than a multi-option menu.
- Run `black src/ tests/ tools/` before suggesting a commit — the user has
  already run `Apply audit fixes and black formatting` (commit `ad9ccd5`) and
  expects formatted output going forward.
