FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace/seg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace/seg
ENV MLFLOW_TRACKING_URI=outputs/mlruns

CMD ["python", "main.py", "--exp_config", "config/experiments/exp_01.yaml"]