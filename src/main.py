import os
import json
import time

import numpy as np
import pandas as pd

import openai

import psycopg2
from psycopg2 import OperationalError

from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

#load_dotenv(dotenv_path="../.env")
load_dotenv()

DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = os.getenv("POSTGRES_PASSWORD")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")

while True:
    try:
        conn = psycopg2.connect(
                            host=DB_HOST,
                            port=DB_PORT,
                            user=DB_USER,
                            password=DB_PASSWORD,
                            dbname=DB_NAME
)
        print("Conectado ao banco de dados!")
        break
    except OperationalError:
        print("Banco de dados ainda não está pronto, tentando novamente...")
        time.sleep(5)  # Aguarda 5 segundos antes de tentar novamente


cur = conn.cursor()
cur.execute("SELECT id, motel_id, session_id, content, remote, created_at FROM message")
rows = cur.fetchall()

cur.close()
conn.close()

df_messages = pd.DataFrame(rows, columns = ['message_id', 'motel_id', 'session_id', 'content', 'remote', 'created_at'])

df = df_messages.copy()
df = df.sort_values(by=['session_id','created_at']).reset_index(drop = True)

df['diff_tempo_entre_mensagens'] = df.groupby(['session_id'])['created_at'].diff()

df['flag'] = 0

df.loc[df['diff_tempo_entre_mensagens'] >= pd.Timedelta("12 hours"), 'flag'] = 1

df['cum_flag'] = df.groupby('session_id')['flag'].cumsum()

aux_unique = df.groupby('session_id')['cum_flag'].unique().sort_index()

aux_nunique = df.groupby('session_id')['cum_flag'].nunique().sort_index()


np.random.seed(10)

for session_id in aux_nunique[aux_nunique>1].index.tolist():

    for cum_flag in aux_unique[aux_unique.index==session_id].values[0].tolist():
        
        new_id = np.random.randint(1, 30)
        ids_existentes = df['session_id'].unique().tolist()
        
        while new_id in ids_existentes:
            new_id = np.random.randint(1, 30)

        df.loc[(df['session_id']==session_id) & (df['cum_flag'] == cum_flag), 'session_id'] = new_id


df['min_created_session'] = df.groupby(['session_id'])['created_at'].transform("min")

df = df.sort_values(by = ['min_created_session', 'created_at']).reset_index(drop = True)

df['prev_remote'] = df.groupby(['session_id'])['remote'].shift()
df['prev_created_at'] = df.groupby(['session_id'])['created_at'].shift()

df['prev_remote'] = df.groupby(['session_id'])['remote'].shift()
df['prev_created_at'] = df.groupby(['session_id'])['created_at'].shift()

mask = ((df['remote'] == False) & (df['prev_remote'] == True))

df['tempo_resposta'] = np.nan

df.loc[mask, 'tempo_resposta'] = df['created_at'] - df['prev_created_at']


openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()


def analisar_conversa(texto_conversa, messages = None, temperature = 0.0, frequency_penalty = 0.0, presence_penalty = 0.0):

    system_prompt = (
        "Você é um analista de conversas especializado em reservas de motel. "
        "As conversas ocorrem entre um cliente (marcado como 'user') e uma chatbot chamada 'Alzira' (marcada como 'Alzira').  "
        "Cada mensagem traz data/hora, bem como o tempo que a chatbot levou para responder a última mensagem do cliente.  "
        "As mensagens estão separadas por linhas em branco e vários hifens.  "

        "Sua tarefa é analisar uma única conversa fornecida e retornar exclusivamente* um objeto JSON com a seguinte estrutura:"

        "{\n"
        '  "satisfaction": <número inteiro de 0 a 10>,\n'
        '  "summary": "<resumo breve e factual da conversa>",\n'
        '  "improvement": "<dicas/sugestões para melhorar a conversa>",\n'
        '  "reserva": "<true ou false>",\n'
        '  "income": "<valor numérico ou 0 caso não haja reserva>",\n'
        '  "date": "<data/hora da reserva em formato ISO 8601 ou null se não houver>"\n'
        "}\n\n"


        "Instrucoes:\n\n"
        "1. Satisfaction (0 a 10). Avalie quão satisfeito o cliente aparenta estar. Baseie-se especialmente em sinais de satisfação ou insatisfação no texto, no impacto de eventuais atrasos nas respostas e se o cliente parou de responder antes de reservar.\n\n"
        "2. Summary. Faca um resumo objetivo e conciso dos principais pontos da conversa: qual era a solicitacão do cliente, quais informações o chatbot forneceu, se houve confirmação de reserva etc. Não especifique o nome do cliente\n\n"
        "3. Improvement. Apresente sugestões para melhorar o atendimento da Alzira (clareza, cordialidade, detalhamento de opções, idioma conversado). Aqui você pode ter mais criatividade.\n\n"
        "4. Reserva (true ou false). Retorne true se houve uma confirmação clara de reserva (quarto escolhido, data agendada). Se houve a geração de link de pagamento, considere que ocorreu a reserva. Se o cliente apenas perguntou ou não concluiu, retorne false.\n\n"
        "5. Income (valor em reais). Se a conversa confirmou um quarto (e possivelmente adicionais), retorne o valor total. Se não houver reserva, retorne 0.\n\n"
        "6. Date. Se a data ou hora da reserva foi confirmada, retorne em formato ISO 8601 (por exemplo, 2025-02-01T14:00:00). Se não houver data agendada, retorne null.\n\n"

        "Muito importante: Retorne apenas o objeto JSON, sem texto adicional, cabeçalhos de cdigo ou comentários. Não invente informações que não estejam na conversa. Se algo não for mencionado, use defaults apropriados (reserva=false, income=0, date=null etc.). Seja coerente com o tom do cliente e os fatos descritos.\n"
    )


    if messages == None or len(messages) == 0:
        messages = [
            {"role": "developer", "content": system_prompt},
        ]


    user_prompt = (
        f"Aqui está a conversa:\n\n{texto_conversa}\n\n"
        "Lembre: Retorne exclusivamente JSON e nada mais. Sem códigos de linguagem natural, sem blocos de código, sem texto adicional. Somente o objeto JSON."
    )

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # ou outro modelo que você use
        messages=messages,
        temperature=temperature,
        presence_penalty = presence_penalty,
        frequency_penalty = frequency_penalty
    )

    #resposta = response.choices[0].message.content
    
    return response, messages


df_conversas = pd.DataFrame()
messages = []

for session_id in df.session_id.unique().tolist():

    df_aux = df.loc[df['session_id']==session_id].sort_values(by='created_at')

    string = ""

    for i, row in df_aux.iterrows():

        if row['remote'] == True:
            individuo = "User"
            string_aux = f"Dia e hora da mensagem: {row['created_at']} \n {individuo}: {row['content']} \n\n {100 * '-'} \n"
        
        elif row['remote'] == False:
            tempo_resposta = row['tempo_resposta']
            individuo = "Alzira"
            string_aux = f"Dia e hora da mensagem: {row['created_at']} \n Tempo de resposta: {tempo_resposta} \n {individuo}: {row['content']}  \n\n {100 * '-'} \n"

        string = string + string_aux

    response, messages = analisar_conversa(string, messages, 0.3, 0.5, 0.5)
    resposta = json.loads(response.choices[0].message.content)
    
    resposta['conversa'] = string
    resposta['session_id'] = session_id

    for key, value in resposta.items():
        resposta[key] = [value]

    df_conversas = pd.concat([df_conversas, pd.DataFrame(resposta)], axis = 0).reset_index(drop = True)


    messages.append({'role':'assistant', "content": response.choices[0].message.content})



df_conversas['id'] = np.arange(1, df_conversas.shape[0] + 1)

df_aux = df.groupby(['session_id'])[['created_at']].min().reset_index()
df_conversas = df_conversas.merge(df_aux, on = 'session_id', how = 'left')


df_conversas['id'] = df_conversas['id'].astype(int)
df_conversas['satisfaction'] = df_conversas['satisfaction'].astype(int)
df_conversas['summary'] = df_conversas['summary'].astype(str)
df_conversas['improvement'] = df_conversas['improvement'].astype(str)
df_conversas['reserva'] = df_conversas['reserva'].astype(int)
df_conversas['income'] = df_conversas['income'].astype(float)
df_conversas['date'] = pd.to_datetime(df_conversas['date'])
df_conversas['conversa'] = df_conversas['conversa'].astype(str)
df_conversas['session_id'] = df_conversas['session_id'].astype(int)

df_analise = df_conversas[['session_id', 'satisfaction', 'summary', 'improvement', 'created_at', 'reserva','income', 'date']].rename(columns = {'date':"data_reserva"})

df_analise['created_at'] = df_analise['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')


while True:
    try:
        conn = psycopg2.connect(
                            host=DB_HOST,
                            port=DB_PORT,
                            user=DB_USER,
                            password=DB_PASSWORD,
                            dbname=DB_NAME
)
        print("Conectado ao banco de dados!")
        break
    except OperationalError:
        print("Banco de dados ainda não está pronto, tentando novamente...")
        time.sleep(5)  # Aguarda 5 segundos antes de tentar novamente

cur = conn.cursor()

session_ids_analysis = df_analise['session_id'].unique()
cur.execute("SELECT id FROM session;")
session_ids_in_session = [row[0] for row in cur.fetchall()]

missing_session_ids = set(session_ids_analysis) - set(session_ids_in_session)

df_aux = df.loc[df['session_id'].isin(missing_session_ids)].sort_values(by='created_at').drop_duplicates(subset=['session_id'],keep='first')[['session_id','motel_id','created_at']]


# Inserir os dados faltantes no banco
for _, row in df_aux.iterrows():
    cur.execute(
        """
        INSERT INTO session (id, motel_id, created_at)
        VALUES (%s, %s, %s)
        """,
        (row['session_id'], row['motel_id'], row['created_at'])
    )

# Confirmar mudanças no banco
conn.commit()

# Query de inserção
sql_insert = """
    INSERT INTO analysis (session_id, satisfaction, summary, improvement, created_at)
    VALUES (%s, %s, %s, %s, %s)
"""

# Converter DataFrame para lista de tuplas e executar inserção
data_to_insert = df_analise[["session_id", "satisfaction", "summary", "improvement", "created_at"]].to_records(index=False).tolist()
cur.executemany(sql_insert, data_to_insert)

# Confirmar mudanças no banco
conn.commit()

cur.close()
conn.close()






























