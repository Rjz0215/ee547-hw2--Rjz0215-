FROM python:3.11-slim
WORKDIR /app
COPY arxiv_server.py /app/
COPY sample_data/ /app/sample_data/
EXPOSE 8080
ENTRYPOINT ["python", "/app/arxiv_server.py"]
CMD ["8080"]