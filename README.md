# ValidationCone [Outdated]

The data was collected on Springfield Avenue on August 31st, 2016, and it includes:
- Rapberry Pi video
- Laser range finder data
- Manual speed labels 

The first car passed at 10:17:56 on Aug 31st, 2016 is a black(brown) saturn SUV. There are in total 79 detections. The 50th and 63rd vehicles are motorcycles. The 68th speed is not clear.

The video preprocessing script is `process_video.sh`. The vision algorithm is packed inside Python script `ValidationCone.py`.

The processed dynamic information is stored as an array in `wheel_dynamics.npy`. The array is three dimensional and can be accessed as `wheel_dynamics[wheel_ID, time, features]`. `wheel_ID` is assigned in the order of time, i.e., the nth passing vehicle has an ID of (n-1). `time` contains 640 instances, corresponding to the 640 horizontal locations that the cener of wheel will take in the 640 by 480 Pi video. Within each instance, the `features` are timestamp (0), horizontal pixel location (1), vertical pixel location (2), and the radius of the wheel in pixel unit (3).

The fourth vehicle (7th and 8th wheels) in the wheel_dynamics.npy matches the first vehicle in the log file.
