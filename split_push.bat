@echo off
echo This script will help you split your push into smaller commits

rem Push code files first (smallest)
git add *.py *.md .gitignore
git commit -m "Add code files and documentation"
git push -u origin main

rem Then push larger files
git add *.pt
git commit -m "Add model files"
git push origin main

rem Finally push any remaining files
git add .
git commit -m "Add remaining files"
git push origin main

echo Done!
