@echo off
title Hugging Face Bucket Sync - Docs
echo ===============================================================
echo Starting Sync: Local Docs --^> Hugging Face PrivateGarden Bucket
echo ===============================================================
echo.
echo Source:      D:\AI_Data\WQB-Consultant-Data\Docs
echo Destination: hf://buckets/teaflower/PrivateGarden-storage/Docs
echo.
echo Calculating chunks and syncing files...
echo.

REM Execute the Hugging Face sync command
hf buckets sync D:/AI_Data/WQB-Consultant-Data/Docs hf://buckets/teaflower/PrivateGarden-storage/Docs

echo.
echo ===============================================================
echo Sync process finished!
echo ===============================================================
pause