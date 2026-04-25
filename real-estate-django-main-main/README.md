# Real Estate Django App

## Start

```bash
docker compose up --build
```

Then pull the AI model (first time only):

```bash
docker compose exec ollama ollama pull llama3.2:3b
```

Open [http://localhost:8000](http://localhost:8000)

## Stop

```bash
docker compose down
```
