# Overview

AI Object Detection w/ Intel Xeon CPU!

First Team to 500 Points Wins!




In this round, you are tasked with levering Intel's Xeon CPU to perform AI inferrence on a Live RTSP camera feed. Your goal is to achieve the fastest possible inference frames per second and as EXTRA CREDIT with the minimume nuber of Xeon cpu CORES.  Note that each Xeon CPU core has a build in AI Accelerator called AMX (Advanced Matrix Extensions).

Try to achieve > 20 inference frames per second ! with ? number of Xeon CPU core(s) !

## Helpful Resources
- **[Link](https://....):** This is a link that helps

## Tips
- tip 1
- tip 2

START Here:  https://app.padme.ai/

| Team   | Username         | Password ---|
| ------ | ---------------- | ----------- |
| Team 1 | team_1@intel.gdc | intel-gdc-2 |
| Team 2 | team_2@intel.gdc | intel-gdc-2 |
| Team 3 | team_3@intel.gdc | intel-gdc-2 |
| Team 4 | team_4@intel.gdc | intel-gdc-2 |
| Team 5 | team_5@intel.gdc | intel-gdc-2 |
| Team 6 | team_6@intel.gdc | intel-gdc-2 |


SSH to NODE 2 (of 3) to determine how many CPU cores are used with the "top" command:

| Team   | ssh command                  | Password            |
| ------ | ---------------------------- | ------------------- |
| Team 1 | ssh abm-admin@192.168.201.12 | troubled-marble-150 |
| Team 2 | ssh abm-admin@192.168.202.12 | troubled-marble-150 |
| Team 3 | ssh abm-admin@192.168.203.12 | troubled-marble-150 |
| Team 4 | ssh abm-admin@192.168.204.12 | troubled-marble-150 |
| Team 5 | ssh abm-admin@192.168.205.12 | troubled-marble-150 |
| Team 6 | ssh abm-admin@192.168.206.12 | troubled-marble-150 |





| TASK                               | Points  | Tip                             |
| ---------------------------------- | --------| -----------------------------   |
| Get LIVE Inference working         | 100     | Use LOGIN, ANY model will do!   |
| Use Intel AMX                      | 100     | Use OpenVINO Framework          |
| Find the fastest Model             | 100     | Try different Models            |
| Get the fastest AI Inference FPS   | 100     | MAX the Average E2E FPS >40     |
| Find the best precision            | 100     | MAX the Average E2E FPS >50     |
