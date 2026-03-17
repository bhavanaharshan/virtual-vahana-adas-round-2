# Virtual Vahana: Autonomous Driving Stack 🚗

This repository contains a modular, closed-loop vehicle intelligence stack developed for **Round-2 of the Student ADAS & Assisted Driving Challenge**. The system is designed to navigate simulated urban environments in CARLA, manage dynamic traffic conditions, and maintain safety through multi-layered fallback mechanisms.

## System Architecture

The stack is strictly modular, divided into four core pipelines to transition from basic driver-assistance to semi-autonomous vehicle behavior:

1. **Perception (`core/perception.py`):** Utilizes **YOLOv8 Nano** to process RGB camera feeds in real-time, detecting dynamic obstacles (vehicles, bicycles, motorcycles) and pedestrians.
2. **Planning (`core/planning.py`):** Interfaces with CARLA's HD Map API to generate forward-looking waypoints, actively filtering out oncoming traffic lanes to prevent routing errors.
3. **Control (`core/control.py`):**  **Lateral Control:** Implements a **Stanley Controller** to compute cross-track and heading errors, keeping the ego-vehicle centered in its lane.
   * **Longitudinal Control:** Uses a **PID Controller** to manage throttle and braking for smooth acceleration to target speeds.
4. **Safety & Arbitration (`core/safety.py`):** An independent Autonomous Emergency Braking (AEB) layer. It calculates depth using bounding box dimensions within a defined "AEB Corridor" and forces an immediate throttle override/full-brake if a collision is imminent.

## Key Features
* **Dynamic Obstacle Handling:** Detects and brakes for slow-moving/stopped vehicles and road obstacles.
* **Pedestrian Safety:** Yields to pedestrians in the vehicle's path with reliable resumption logic.
* **Dynamic Weather Resilience:** Includes a hot-swappable weather system (Clear, Sunset, Rain) to stress-test the perception module under varying visual conditions.

## Prerequisites & Hardware 
This stack was built and tested on Ubuntu with an **Acer Nitro 16 (NVIDIA RTX 4050, 8GB RAM)**. To ensure smooth operation alongside the deep learning models, the CARLA simulator is optimized to run with the `-quality-level=Low` flag.

### Software Requirements
* CARLA Simulator 0.9.16
* Python 3.12+
* Dependencies: `pip install -r requirements.txt`

## Running the Stack
1. **Start the CARLA Server:**
   ```bash
   ./CarlaUE4.sh -quality-level=Low

2.**Activate the environment and run:**
    ```bash
       source /home/bhavana-ph/Virtual\ Vahana/vahana/bin/activate
       python main.py
    
3. **Controls**
Press w to cycle through dynamic weather conditions
Press q to safely terminate stack and clean up simulator actors



