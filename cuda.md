If this seems too complicated you can change Max to use google for speech-to-text instead in the .env

1. download the cuda toolkit: `https://developer.nvidia.com/cuda-downloads`
2. download cudann: `https://developer.nvidia.com/cudnn-downloads`
3. unzip cudann and copy all of the .dll files
4. paste the .dll files in the toolkit\cuda\bin folder (for example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`)
5. now we need to add those .dll's to your PATH, to do this hit the windows key and type "enviro",
6. select "edit the system environment variables"
7. select button on the bottom right "Environment Variables"
8. in the lower window "System variables" find and select "Path"
9. select "Edit"
10. select "Browse"
11. browse to the same location as step 4, where you just put the .dll files
12. then select a ok a bunch of times and close out the menu