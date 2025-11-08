name: Update Bears FPL data

on:
  schedule:
    - cron: "0 */6 * * *"   # every 6 hours
  workflow_dispatch: {}      # allows manual run

jobs:
  fetch-fpl-data:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install requests

      - name: Run update script
        run: python update_bears_data.py

      - name: Commit and push changes
        run: |
          git config user.name "github-actions"
          git config user.email "github-actions@users.noreply.github.com"
          git add public/bootstrap.json public/fixtures.json public/entries/*.json || echo "Nothing to add"
          git commit -m "Auto-update FPL data" || echo "No changes to commit"
          git push || echo "No changes to push"
