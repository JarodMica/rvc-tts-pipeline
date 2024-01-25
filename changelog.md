# Changelogs

# 1/24/2023
- Made a change to default to "cpu" if CUDA or MPS is not detected, enabling CPU inference

# 10/7/2023
Needed to make an adjustment so that the lightweight branch could be installed as a package that is not reliant on edittable mode (for distribution purposes)
- Put rvc_infer module into rvc_pip, so to call rvc_infer, it can be found (after installing the branch) by ```rvc_pipe.rvc_infer```
    - This makes the package easier to move around

# 10/1/2023
I decided to introduce a lightweight branch that just uses the RVC modules itself instead of instantiating that stuff in rvc_infer.  Reduces the code overall and makes things clearer on what is happening.
- NOTES:
    - There is a master branch of tts_pipeline which is still using the rvc package installed as an editable package
    - There is also a lightweight branch of tts_pipeline (which is this one) that will use a downloadable package instead of an installable package which contains all of the modules needed for rvc
- Added a "verbose" option

The purpose of the **lightweight branch** is to be able to use the rvc_infer as long as the rvc folder is located in the same location of the script being ran (preferabbly parent directory).  This cuts down on the amount of refactoring that I would need to do in order to make rvc an installable package (without having it in editabble mode), while maintaining the ability to easily update it in the future in case new functionality is added.  Maintaining an installable package would be too much for myself.
- Deleted duplicate functions in the rvc_infer module
- Made it compatible with my modified version of the latest rvc update as of 10/1/2023
    - Still requires modifications to the rvc repository due to imports, but much less than if I were to make it an installable package

## 8/18/2023
- Added a return path in rvc_convert
- Added mps for Mac for device configuration (untested as I don't have a Mac)

