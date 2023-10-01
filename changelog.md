# Changelogs

# 10/1/2023
I decided to introduce a lightweight branch that just uses the RVC modules itself instead of instantiating that stuff in rvc_infer.  

The purpose of the **lightweight branch** is to be able to use the rvc_infer as long as the rvc folder is located in the same location of the script being ran (preferabbly parent directory).  This cuts down on the amount of refactoring that I would need to do in order to make rvc an installable package (without having it in editabble mode), while maintaining the ability to easily update it in the future in case new functionality is added.  Maintaining an installable package would be too much for myself.
- Deleted duplicate functions in the rvc_infer module
- Made it compatible with my modified version of the latest rvc update as of 10/1/2023
    - Still requires modifications to the rvc repository due to imports, but much less than if I were to make it an installable package

## 8/18/2023
- Added a return path in rvc_convert
- Added mps for Mac for device configuration (untested as I don't have a Mac)

