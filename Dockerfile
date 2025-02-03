# Usar uma imagem base do Python
FROM python:3.11-slim

# Configurar o diretório de trabalho no container
WORKDIR /app

# Copiar os arquivos necessários para o container
COPY requirements.txt requirements.txt
COPY src/ src/
COPY .env .env

# Instalar dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Definir o comando padrão para rodar ao iniciar o container
CMD ["python", "src/main.py"]