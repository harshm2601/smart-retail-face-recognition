@echo off
echo Installing Git LFS...
git lfs install

echo Tracking large files with Git LFS...
git lfs track "*.pt"
git lfs track "*.db" 
git lfs track "*.mp4"

echo Committing changes...
git add .gitattributes
git commit -m "Configure Git LFS"

echo Now try adding and pushing your files again
echo git add .
echo git commit -m "Your message"
echo git push -u origin main
