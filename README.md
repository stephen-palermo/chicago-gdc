# AI Object Detection w/ Intel CPU and Google GDC

**First team to 500 points wins!**

## Object Detection Challenge

### 1. Setup

**Goal:** Get familiar with the GUI and CPU.

- **(a) SSH into node2 and check CPU for AMX:** `lscpu | grep amx`
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

### 4. Expected Results

**Setup — AMX check:** Running `lscpu | grep amx` should list the AMX feature flags, confirming the CPU supports Advanced Matrix Extensions:

```console
$ lscpu | grep amx
Flags: ... amx_bf16 avx512_fp16 amx_tile amx_int8 ...
```

Look for **`amx_bf16`**, **`amx_tile`**, and **`amx_int8`** in the output.

**Setup — CPU utilization:** Running `btop` should display per-core CPU utilization, frequency, and temperatures. On this platform you should see a **Xeon Gold 6438N** with 32 cores (C0–C31):

```console
16:19:10                                                    2000ms
Xeon Gold 6438N                                             2.3 GHz
CPU [||||||||||||||||||||||||||||||||||||||||           53%]  45°C
C0  54%  42°C   C7  54%  43°C   C14 54%  43°C   C21 53%  42°C  C28 54% 43°C
C1  53%  39°C   C8  53%  43°C   C15 54%  43°C   C22 52%  40°C  C29 52% 42°C
C2  52%  43°C   C9  53%  42°C   C16 54%  43°C   C23 53%  40°C  C30 53% 41°C
C3  53%  40°C   C10 51%  43°C   C17 54%  42°C   C24 54%  41°C  C31 52% 43°C
C4  53%  41°C   C11 53%  41°C   C18 55%  44°C   C25 54%  41°C
C5  53%  44°C   C12 53%  40°C   C19 53%  40°C   C26 54%  41°C
C6  53%  43°C   C13 52%  40°C   C20 53%  42°C   C27 52%  42°C  LAV: 19.5 21.5 14.7
```

**Run Inference — live overlay:** Once inference is running, the video feed shows live object detections with an overlay of runtime metrics:

```text
Time: 2026-09-15T16:20:20 || Frame: 3858 || Elapsed: 6m 54s 980.00ms
Use Case: dwell/time || Model: [Demo] - DFINE-nano - mscoco80 - (640,640) - FP16
Video FPS (Source / Target / Processed): 30.00 / 30.00 / 9.30 || AI FPS (Now / Avg): 8.79 / 9.67
Framework: ONNX (CPU) || Precision: FP16 || CPU (%): 1601.4 || RAM (MB): 420.5 || Watts: 287.6
Mask 1: Occupancy 1 || Avg. dwell time (sec.) 155.0
```

## Overview

In this round, you are tasked with leveraging Intel's Xeon CPU to perform AI inference on a live RTSP camera feed. Your goal is to achieve the fastest possible inference frames per second (FPS) and, as **extra credit**, to do so with the minimum number of Xeon CPU cores.

Each Xeon CPU core has a built-in AI accelerator called **AMX (Advanced Matrix Extensions)**.

> **Goal:** Achieve **> 20 inference FPS** using as few Xeon CPU cores as possible.

## Login to Object Detection GUI

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

SSH into **node2** (the container that will run the inference) with the following credentials:

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
