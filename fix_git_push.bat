@echo off
git config --global http.postBuffer 524288000
git config --global http.maxRequestBuffer 100M
git config --global core.compression 9
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300

echo Git configurations updated for large pushes.
echo Now try pushing again:
echo git push -u origin main
