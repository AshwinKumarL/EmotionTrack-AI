# Project-Scoped Agent Rules

## Routine Actions Approval Policy
Automatically approve routine development actions without prompting the user for confirmation every time.

**Allowed Auto-Approve Actions:**
*   Creating new source files
*   Editing existing project files
*   Running Python scripts
*   Running pytest
*   Installing project dependencies listed in `requirements.txt`
*   Creating documentation files
*   Creating reports
*   Reading project files
*   Reading dataset metadata
*   Generating logs
*   Running preprocessing scripts
*   Creating CSV and JSON metadata files

**Require Manual Confirmation for:**
*   Deleting files or folders
*   Renaming or moving major project directories
*   Overwriting raw datasets
*   Modifying Git history (force push, reset, rebase, delete branches, etc.)
*   Removing installed packages
*   Executing destructive shell commands
*   Performing actions outside the project directory
*   Requiring secrets, credentials, or external accounts
