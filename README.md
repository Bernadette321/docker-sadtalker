# SadTalker Containerized Deployment

## Structure

```
app/
  main.py
  sadtalker/
    inference.py
requirements.txt
Dockerfile
```

## Usage

1. Build Docker image:
   ```
   docker build -t sadtalker_app .
   ```

2. Run container:
   ```
   docker run -p 8000:8000 sadtalker_app
   ```

3. Access the API docs at http://localhost:8000/docs

## Next Steps

- Implement actual SadTalker inference logic in `app/sadtalker/inference.py`.
- Add model weights and any additional configurations.