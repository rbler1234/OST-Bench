<br>
<p align="center">
<h1 align="center"><strong>OST-Bench: An online spatial understanding benchmark</strong></h1>

</p>
</p>

<div id="top" align="center">



</div>



![demo](assets/benchmark_samples.png "demo")

<!-- contents with emoji -->

## 📋 Contents

1. [About](#-about)
2. [Getting Started](#-getting-started)
3. [Evaluation](#-evaluation)
4. [Leaderboard](#-leaderboard)


## 🏠 About

<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/teaser_new.png" alt="Dialogue_Teaser" width=100% >
</div>

Recent advances in multimodal large language models (MLLMs) have shown remarkable capabilities
in integrating vision and language for complex reasoning. While most existing benchmarks evaluate models under offline settings with a fixed set of pre-recorded inputs, we introduce OST-Bench, a benchmark designed to evaluate Online Spatio-Temporal understanding from the perspective of an agent actively exploring a scene.  The “Online” aspect emphasizes the need to process and reason over incrementally acquired observations, while the “Spatio-Temporal” component requires integrating current visual inputs with historical memory to support dynamic spatial reasoning. OST-Bench better reflects the challenges of real-world embodied perception. Built on an efficient data collection pipeline, OST-Bench consists of 1.4k scenes and 10k question-answer pairs collected from ScanNet, Matterport3D, and ARKitScenes. We evaluate several leading MLLMs on OST-Bench and observe a significant drop in performance on tasks requiring complex spatio-temporal reasoning, with accuracy degrading as the exploration horizon increases and memory grows under the online setting. Through further experimental analysis, we identify common error patterns across models and find that complex spatial reasoning demands and long-term memory retrieval requirements lead to a significant drop in model performance, highlighting the core challenges that must be addressed to improve online embodied reasoning. Our dataset and benchmark will be publicly released to foster further research and development in the field.

## 🚀 Getting Started

- ### Installation

1. Clone Github repo.

   ```shell
   git clone git@github.com:rbler1234/MMScan.git
   cd MMScan
   ```

2. Install requirements.


   ```shell
   conda activate your_env_name
   pip install -r requirements.txt
   ```

   *Note:* If you want to evaluate open-source models, you need to set up their corresponding environments.

- ### Data Preparation

1. Download the dataset of OST-Bench from [kaggle](https://www.kaggle.com/datasets/jinglilin/ost-bench/).
 
2. Unzip the image files and the json file, place them as followed:

## 👓 Evaluation


### Proprietary Model (GPT, Gemini, Claude)



- #### Setup 

  Before this, you need to fill in your api_key in `models/utils/openai_api.py`.

  ```bash
  API_keys = {
    'gpt': "your_api_key",
    'claude':"your_api_key",
    'gemini':"your_api_key"
   }
  ```

- #### Inference
   We provide unified inference scripts for all proprietary models.
  ```bash
  cd models
  python proprietary_baseline.py --model_name gpt4o --anno_json_path /path/to/anno  --image_root /path/to/image --save_root /path/to/save
  ```

- #### Evaluator
   Use our OST evaluator to get the results.
   ```bash
  cd evaluation
  python OST_evaluator.py --result_dir /path/to/save
  ```

### Open-source Models(QwenVL-2.5, InternVL-2.5)
- #### Setup 

  Follow the [Quickstart of QwenVL](https://github.com/QwenLM/Qwen2.5-VL) and [Quickstart of InternVL](https://internvl.readthedocs.io/en/latest/internvl2.5/quick_start.html) to set up the required environment and download the ckpts.

  

- #### Inference
  ```bash
  cd models
  python InternVL_baseline.py/QwenVL_basline.py --model_path /path/to/ckpt --anno_json_path /path/to/anno  --image_root /path/to/image --save_root /path/to/save
  ```

- #### Evaluator
   Use our OST evaluator to get the results.
   ```bash
  cd evaluation
  python OST_evaluator.py --result_dir /path/to/save
  ```




## 🏆 Leaderboard 




| Methods                    | Agent State |       |             |       | Agent Visible Info |       |          |           |       | Agent-object Spatial  |       |       |          |       |       |
|----------------------------|:-----------:|:-----:|:-----------:|:-----:|:------------------:|:-----:|:--------:|:---------:|:-----:|:---------------------:|:-----:|:-----:|:--------:|:-----:|:-----:|
|                            |   Position  |       | Orientation |       |      Existence     |       | Quantity | Diversity | Order |       Direction       |       |       | Distance |       |       |
|                            |     JUD.    |  EST. |     JUD.    |  EST. |        JUD.        | TEMP. |   CNT.   |    JUD.   |  JUD. |          JUD.         | TEMP. |  EST. |   JUD.   | TEMP. |  EST. |
| Proprietary                |             |       |             |       |                    |       |          |           |       |                       |       |       |          |       |       |
| Claude-3.5-Sonnet          |    55.4     | 32.5  |    54.0     | 26.4  |        79.0        | 61.0  |   51.8   |   75.9    | 59.7  |         43.8          | 15.4  | 22.5  |   47.7   | 55.9  | 19.9  |
| Gemini-2.0-Flash           |    56.4     | 24.4  |    47.2     | 30.4  |        85.0        | 66.9  |   51.3   |   70.5    | 56.4  |         43.4          | 16.7  | 23.9  |   48.5   | 45.3  | 28.6  |
| Gemini-2.0-Flash(thinking) |    57.1     | 24.3  |    59.2     | 26.6  |        81.5        | 75.8  |   52.8   |   69.6    | 62.8  |         51.7          | 34.4  | 28.2  |   47.4   | 57.7  | 23.5  |
| GPT-4o                     |    59.9     | 20.7  |    50.0     | 27.0  |        84.2        | 71.8  |   50.4   |   74.6    | 58.1  |         45.1          | 19.7  | 23.6  |   44.2   | 51.5  | 22.6  |
| GPT-4.1                    |    69.3     | 30.1  |    59.9     | 28.4  |        84.1        | 77.1  |   53.4   |   78.6    | 66.2  |         52.4          | 18.4  | 23.3  |   46.8   | 50.9  | 23.9  |
| Open-source                |             |       |             |       |                    |       |          |           |       |                       |       |       |          |       |       |
| InternVL-2.5-8B            |    54.2     | 24.5  |    52.8     | 38.6  |        81.3        | 18.1  |   45.0   |   59.4    | 43.3  |         35.9          | 10.3  | 21.7  |   43.4   | 27.5  | 27.1  |
| InternVL-2.5-38B           |    64.6     | 31.3  |    56.5     | 34.0  |        86.7        | 70.6  |   50.6   |   73.9    | 53.2  |         42.4          | 17.8  | 28.4  |   44.5   | 44.0  | 29.0  |
| InternVL-2.5-78B           |    61.7     | 33.5  |    51.8     | 34.0  |        85.2        | 70.8  |   54.0   |   77.7    | 50.5  |         42.6          | 20.7  | 17.6  |   46.5   | 43.6  | 20.9  |
| QwenVL-2.5-8B              |    56.1     | 13.6  |    53.3     | 37.3  |        76.3        | 32.4  |   48.2   |   56.9    | 31.7  |         41.2          | 28.2  | 13.2  |   46.0   | 47.5  | 18.6  |
| QwenVL-2.5-32B             |    59.2     | 29.6  |    55.4     | 37.8  |        81.0        | 61.5  |   47.6   |   73.6    | 40.3  |         40.9          | 23.0  | 27.2  |   48.8   | 44.5  | 19.5  |
| QwenVL-2.5-72B             |    62.2     | 26.2  |    50.5     | 32.2  |        79.9        | 61.0  |   45.0   |   74.6    | 30.4  |         43.2          | 15.6  |  5.7  |   44.9   | 44.3  | 19.1  |
| LLaVA-Video-7B             |    53.3     | 25.2  |    50.6     | 10.9  |        85.0        | 31.0  |   50.8   |   59.7    | 38.3  |         36.0          | 31.0  | 16.4  |   38.8   | 40.8  |  9.7  |
| LLaVA-Video-72B            |    50.6     | 19.5  |    50.8     | 38.8  |        84.3        | 37.1  |   30.8   |   70.0    | 45.9  |         40.3          | 26.2  | 23.1  |   44.2   | 48.8  | 23.8  |
| LLaVA-Onevision-7B         |    51.8     | 10.0  |    46.3     |  5.8  |        84.9        | 34.7  |   54.6   |   49.2    | 30.8  |         35.0          | 30.7  | 40.2  |   41.5   | 36.9  | 19.8  |
| LLaVA-Onevision-7B         |    55.8     | 18.8  |    49.3     | 33.4  |        85.0        | 46.6  |   37.2   |   71.3    | 48.8  |         38.4          | 21.3  | 27.1  |   50.9   | 50.2  | 29.2  |
| Baseline                   |             |       |             |       |                    |       |          |           |       |                       |       |       |          |       |       |
| Human-Level                |     93.2    |  58.9 |     92.8    |  54.4 |        95.7        |  94.7 |   91.3   |    94.4   |  90.9 |          90.5         |  93.3 |  54.3 |   93.4   |  94.5 |  60.1 |
| Chance-Level               |     50.0    |  37.8 |     50.0    |  39.3 |        50.0        |  29.1 |   25.0   |    33.0   |  25.0 |          36.0         |  33.2 |  47.6 |   36.0   |  31.2 |  30.3 |
|                            |             |       |             |       |                    |       |          |           |       |                       |       |       |          |       |       |
|                            |             |       |             |       |                    |       |          |           |       |                       |       |       |          |       |       |

## 📝 TODO List

- \[ \] Full-code release.

