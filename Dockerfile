
# Stage 1: Frontend build
FROM node:18 as frontend-build
WORKDIR /frontend
COPY ./frontend/package.json ./frontend/package-lock.json ./
RUN npm install
COPY ./frontend ./
RUN npm run build

# Stage 2: Backend setup
FROM python:latest
WORKDIR /app

# Copy backend requirements and install them
COPY .//requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY ./backend /app

# Copy frontend build output to the backend static directory
COPY --from=frontend-build /frontend/build /app/static/

# Expose the port (Heroku uses port 5000 for web apps)
EXPOSE 5000

# Command to run FastAPI backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
