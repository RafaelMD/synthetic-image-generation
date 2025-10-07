# 🖼️ Synthetic Image Generation

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![Google GenAI](https://img.shields.io/badge/Google-GenAI-orange)](https://ai.google.dev/)  

This project automates the generation of **synthetic images with AI** by adding custom entities (e.g., *dogs*, *cows*, etc.) into highway images, applying **data augmentation**, and filtering results with an **AI judge** to ensure quality.  
It also produces a **report** summarizing the number of images processed, successes, failures, discarded outputs, and augmented results.

---

## 📂 Project Structure
```plaintext
├── input_images/ # Folder with original images to process
├── output_images/ # Folder where accepted + augmented images are saved
│ └── <entity>/ # Entity-specific output folder (e.g., dogs, cows)
├── discarded_images/ # AI-rejected images (organized by entity)
├── .env # API key configuration
├── requirements.txt # Project dependencies
├── synthetic_crew/ # CrewAI orchestration package
│ ├── agents.py # Crew agent role definitions
│ ├── pipeline.py # End-to-end crew pipeline
│ ├── tasks.py # Task prompts
│ └── tools.py # Gemini-powered tools
├── main.py # CLI entry-point that runs the crew
└── README.md # Documentation
```

---

## ⚙️ How It Works

1. **Load environment & arguments**  
   - Reads `.env` for `API_KEY`  
   - CLI args define entity, input/output folders, and number of contexts.

2. **Context Analysis (Gemini AI)**  
   - Analyzes each input image and generates JSON contexts describing **where the entity could be placed**.

3. **CrewAI Orchestration**
   - Three agents coordinate the workflow:
     - **Context Analyst** reads the image and proposes insertion scenarios.
     - **Entity Image Generator** calls Gemini to place the entity following a chosen context.
     - **Quality Judge** validates the realism of the generated entity.
   - Each agent operates through dedicated tools that wrap the Gemini APIs and return structured JSON.

4. **Data Augmentation & Reporting**
   - Accepted images are augmented (horizontal flip) and saved next to their originals.
   - The pipeline keeps per-image context metadata, counts successes/failures/discards and writes a summary `report.json`.

### 👥 CrewAI Agents

| Agent | Responsibility | Key Tool |
| ----- | -------------- | -------- |
| **Context Analyst** | Reads the original image and proposes up to `context_limit` scenarios for the requested entity. | `analyze_image_contexts` |
| **Entity Image Generator** | Calls Gemini with the chosen context to synthesize the entity inside the scene. | `generate_entity_image` |
| **Quality Judge** | Verifies that the generated entity looks natural, rejecting unrealistic results. | `judge_generated_image` |

Each agent has a focused goal and operates sequentially through the `SyntheticImageGenerationCrew` orchestrator.

---

## 🖥️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/your-repo/synthetic-image-generation.git
cd synthetic-image-generation
pip install -r requirements.txt
```
Create a .env file in the root directory:
```bash
API_KEY=your_google_genai_api_key
```

**Usage**

Run the script with:
```bash
python main.py -e <entity> [-c CONTEXT_LIMIT] [-i INPUT_FOLDER] [-o OUTPUT_FOLDER] [-d DISCARD_FOLDER]
```

The CLI bootstraps the CrewAI agents, runs the sequential workflow (context discovery → image generation → judging), and falls back to the raw tool calls if the crew layer is unavailable so the pipeline can still complete.

Arguments

| Argument               | Description                              | Default            | Required |
| ---------------------- | ---------------------------------------- | ------------------ | -------- |
| `-e, --entity`         | Entity to insert (e.g., "dog", "cow")    | —                  | ✅ Yes    |
| `-c, --context_limit`  | Number of contexts to generate per image | `3`                | ❌ No     |
| `-i, --input_folder`   | Folder with input images                 | `input_images`     | ❌ No     |
| `-o, --output_folder`  | Folder for generated outputs             | `output_images`    | ❌ No     |
| `-d, --discard_folder` | Folder for AI-discarded results          | `discarded_images` | ❌ No     |

**Example**
```bash
python main.py -e dog -c 2 -i ./my_highways -o ./results -d ./bad_outputs
```
- Adds dogs into each highway image
- Generates up to 2 contexts per image
- Saves results in ./results/dog/
- Discards bad images into ./bad_outputs/dog/

**Report Example**

After execution, a report.json file is saved in the output entity folder:
```bash
{
  "entity": "dog",
  "total_images": 10,
  "api_success": 18,
  "api_failures": 2,
  "augmented_images": 9,
  "discarded": 4,
  "contexts": {
    "highway1.jpg": {
      "1": "dog standing on the roadside",
      "2": "dog in the middle of the road"
    }
  },
  "processing_time": "0h 3m 27s"
}
```
