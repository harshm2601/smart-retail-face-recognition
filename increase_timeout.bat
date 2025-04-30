@echo off
git config --global http.postBuffer 524288000
git config --global http.receivepack 0
git config --global http.timeout 300
git config http.version HTTP/1.1

echo Git timeout settings increased.
echo Now try pushing again:
echo git push -u origin main
