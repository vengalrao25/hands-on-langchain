# Docker Instructions

## Prerequisites

Make sure Docker is installed on your machine. You can verify by running:

```bash
docker --version
```

---

## Step 1 — Add your API Key

Copy the example env file and add your OpenAI key:

```bash
cp .env.example .env
```

Open `.env` and set your key:

```
OPENAI_API_KEY=your_key_here
```

---

## Step 2 — Build Image & Start Container (Once only)

This builds the image and starts a long-running container in the background. You only need to do this once (or whenever you change the Dockerfile).

```bash
./run.sh
```

- Builds the image (`langchain-exp`)
- Starts the container in the background and keeps it alive
- API key is automatically loaded from `.env`

---

## Step 3 — Run Any Python File

Once the container is running, execute any Python file instantly:

```bash
./start.sh main.py
./start.sh experiment.py
```

- No startup delay — container is already running
- Any changes you make to your Python files are picked up immediately
- All print statements and logs appear directly in your terminal

---

## Installed Packages

| Package | What it does |
|---------|-------------|
| `langchain` | Core framework — chains, agents, memory |
| `langchain-openai` | OpenAI integration (ChatOpenAI, embeddings) |
| `langchain-core` | Base interfaces — prompts, runnables, output parsers |

To add more packages, update the `Dockerfile`:

```dockerfile
RUN pip install langchain langchain-openai langchain-core <new-package>
```

Then rebuild by running `./run.sh` again.

---

## How to Stop the Container

```bash
docker stop langchain-container
```

---

## How to Start it Again (without rebuilding)

```bash
docker start langchain-container
```

---

## How to Remove

### Remove the container

```bash
docker rm langchain-container
```

### Remove the image (frees up disk space)

```bash
docker rmi langchain-exp
```

### Remove everything at once

```bash
docker stop langchain-container && docker rm langchain-container && docker rmi langchain-exp
```

---

## When to rebuild (run `./run.sh` again)

| Change made | Action needed |
|-------------|---------------|
| Changed a `.py` file | Nothing — just `./start.sh` |
| Added a new package to `Dockerfile` | `./run.sh` — rebuild required |
| First time setup | `./run.sh` |

---

## Quick Reference

| Task                        | Command                                  |
|-----------------------------|------------------------------------------|
| Build & start container     | `./run.sh`                               |
| Run a Python file           | `./start.sh <filename.py>`               |
| List running containers     | `docker ps`                              |
| Stop container              | `docker stop langchain-container`        |
| Start container again       | `docker start langchain-container`       |
| Remove container            | `docker rm langchain-container`          |
| Remove image                | `docker rmi langchain-exp`               |
