# AI Object Detection with Intel Xeon CPU

**First team to 500 points wins!**

## Object Detection Challenge

### 1. Setup

**Goal:** Get familiar with the GUI and CPU.

- **(a) Check CPU for AMX:** `lscpu | grep amx`
- **(b) Login to the app** and view the live RTSP video.
- **(c) Check CPU utilization:** `btop`

### 2. Run Inference

**Goal:** Start live inference → get 100 points.

- **(a)** Start inference.
- **(b)** Show inference.

### 3. Optimize

**Goal:** Optimize the AI model / framework / precision — get up to 400 points.

- **(a)** Change the AI framework.
- **(b)** Change the model.
- **(c)** Change the precision.
- **(d)** Minimize CPU utilization.
- **(e)** Maximize E2E FPS.

## Overview

In this round, you are tasked with leveraging Intel's Xeon CPU to perform AI inference on a live RTSP camera feed. Your goal is to achieve the fastest possible inference frames per second (FPS) and, as **extra credit**, to do so with the minimum number of Xeon CPU cores.

Each Xeon CPU core has a built-in AI accelerator called **AMX (Advanced Matrix Extensions)**.

> **Goal:** Achieve **> 20 inference FPS** using as few Xeon CPU cores as possible.

## Getting Started

Start here: [https://app.padme.ai/](https://app.padme.ai/)

### Team Logins

| Team   | Username         | Password    |
| ------ | ---------------- | ----------- |
| Team 1 | team_1@intel.gdc | intel-gdc-2 |
| Team 2 | team_2@intel.gdc | intel-gdc-2 |
| Team 3 | team_3@intel.gdc | intel-gdc-2 |
| Team 4 | team_4@intel.gdc | intel-gdc-2 |
| Team 5 | team_5@intel.gdc | intel-gdc-2 |
| Team 6 | team_6@intel.gdc | intel-gdc-2 |

### Monitoring CPU Core Usage

SSH to **Node 2 (of 3)** and run the `top` command to determine how many CPU cores are being used.

| Team   | SSH Command                  | Password            |
| ------ | ---------------------------- | ------------------- |
| Team 1 | ssh abm-admin@192.168.201.12 | troubled-marble-150 |
| Team 2 | ssh abm-admin@192.168.202.12 | troubled-marble-150 |
| Team 3 | ssh abm-admin@192.168.203.12 | troubled-marble-150 |
| Team 4 | ssh abm-admin@192.168.204.12 | troubled-marble-150 |
| Team 5 | ssh abm-admin@192.168.205.12 | troubled-marble-150 |
| Team 6 | ssh abm-admin@192.168.206.12 | troubled-marble-150 |

## Scoring

| Task                             | Points | Tip                           |
| -------------------------------- | ------ | ----------------------------- |
| Get LIVE inference working       | 100    | Use LOGIN, any model will do! |
| Use Intel AMX                    | 100    | Use the OpenVINO framework    |
| Find the fastest model           | 100    | Try different models          |
| Get the fastest AI inference FPS | 100    | Max the Average E2E FPS > 40  |
| Find the best precision          | 100    | Max the Average E2E FPS > 50  |

## Tips

- Use the **OpenVINO** framework to take advantage of Intel AMX.
- Experiment with different models to find the fastest one.
- Try lower-precision formats (e.g., INT8, BF16) to boost throughput.
- Monitor core usage with `top` to optimize for the extra-credit core count.

## Helpful Resources

- **[OpenVINO Toolkit](https://docs.openvino.ai/):** Intel's toolkit for optimizing and deploying AI inference.
- **[Intel AMX Overview](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/advanced-matrix-extensions/overview.html):** Learn more about Advanced Matrix Extensions.

## Technical Overview: Intel AMX for AI Inference

**Intel Advanced Matrix Extensions (Intel AMX)** is a built-in AI acceleration engine integrated into every core of 4th Gen Intel Xeon Scalable processors (Sapphire Rapids) and later generations. It is designed to dramatically speed up the matrix-multiplication operations that dominate deep learning inference and training workloads — without requiring a discrete accelerator or GPU.

### How It Works

AMX introduces a new x86 instruction set and hardware architecture built around two key components:

- **Tiles:** A set of eight new 2D registers, each 1 KB in size, that hold large blocks (tiles) of matrix data directly on the CPU core.
- **Tile Matrix Multiply (TMUL):** A dedicated accelerator engine that performs matrix multiply-accumulate operations on the data held in the tiles.

By keeping large matrix operands in these dedicated registers and processing them with the TMUL unit, AMX minimizes data movement and executes many multiply-accumulate operations per instruction. This delivers a significant performance-per-cycle improvement over previous SIMD approaches such as Intel AVX-512.

### Supported Data Types

AMX is optimized for the reduced-precision data types most commonly used in AI inference:

- **BF16 (bfloat16):** Balances performance and accuracy, ideal for both training and inference.
- **INT8 (8-bit integer):** Maximizes throughput for quantized inference workloads where lower precision is acceptable.

### Why It Matters for This Challenge

- **No GPU required:** AMX enables high-performance AI inference directly on the CPU, simplifying deployment and reducing system cost.
- **Higher throughput per core:** Because AMX is embedded in each core, more inference work is accomplished per core, which helps maximize FPS while minimizing the number of cores used — directly rewarding the extra-credit goal.
- **Software support:** Frameworks such as **OpenVINO**, PyTorch, and TensorFlow can automatically dispatch supported operations to AMX. Using OpenVINO with BF16 or INT8 models is the most direct way to unlock AMX acceleration in this challenge.

### Key Takeaway

To achieve the best inference FPS with the fewest cores, run an **INT8- or BF16-optimized model** through **OpenVINO**, which will automatically leverage the AMX engine built into each Xeon core.
