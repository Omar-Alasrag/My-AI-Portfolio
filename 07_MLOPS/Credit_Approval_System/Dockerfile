FROM astral/uv:python3.13-trixie-slim


WORKDIR /app


COPY pyproject.toml uv.lock  ./

RUN uv sync --frozen --no-install-project


COPY . .

RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]