# TPMD Agent — RAG com LangGraph

Agente de conhecimento sobre tráfego pago e marketing digital, usando LangGraph + FAISS + Gemini.

## Estrutura do Projeto

```
tpmd-agent/
├── data/
│   └── docs/              # Markdowns gerados por tópico (criados pelo script 01)
├── vectorstore/           # Índice FAISS (criado pelo script 02)
├── src/
│   ├── ingest/
│   │   ├── vtt_parser.py       # Parser de legendas .vtt
│   │   ├── pdf_extractor.py    # Extrator de texto de PDFs
│   │   └── doc_generator.py    # Gerador de markdown via GPT-4.1-mini
│   ├── rag/
│   │   └── indexer.py          # Criação e carregamento do índice FAISS
│   └── agent/
│       ├── state.py            # Estado do grafo LangGraph
│       ├── nodes.py            # Nodes: retrieve e generate
│       └── graph.py            # Construção do grafo
├── scripts/
│   ├── 01_generate_docs.py     # Passo 1: gerar documentação Markdown
│   └── 02_build_index.py       # Passo 2: construir índice vetorial
├── app.py                      # Interface Streamlit
├── requirements.txt
└── .env.example
```

## Requisitos

- Python **3.10 ou superior** (recomendado: 3.13)
- Chave de API Gemini (gratuita) ou OpenAI
- Pasta com os arquivos de conteúdo (`.vtt` e/ou `.pdf`) — necessária apenas para reprocessar do zero (passos 3 e 4)

## Setup

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar variáveis de ambiente

```bash
copy .env.example .env
```

Edite o `.env` e configure sua `OPENAI_API_KEY`. Você pode gerar uma chave em [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

### 3. Rodar o agente

```bash
streamlit run app.py
```

Ou simplesmente dê dois cliques no arquivo `run.bat` na raiz do projeto.

**Dica:** crie um atalho do `run.bat` na área de trabalho para acesso rápido:

1. Clique com o botão direito no `run.bat`
2. Selecione **Enviar para → Área de trabalho (criar atalho)**
3. Renomeie o atalho para `Agente`

---

## Reprocessar o conteúdo do zero (opcional)

Necessário apenas se quiser regenerar a documentação e o índice a partir dos arquivos originais do curso.

### Passo 1 — Gerar documentação Markdown

Processa os arquivos `.vtt` e `.pdf` do curso e gera um documento Markdown por lição.

```bash
python scripts/01_generate_docs.py
```

Para retomar de onde parou (sem reprocessar lições já concluídas):

```bash
python scripts/01_generate_docs.py --resume
```

### Passo 2 — Construir índice vetorial FAISS

```bash
python scripts/02_build_index.py
```

## Pipeline de Processamento

```
Pasta do curso
    └── [módulo]/[lição]/
            ├── 1. Aula.pt_br.vtt  ──→ texto limpo
            └── slides.pdf         ──→ texto extraído
                        │
                        ▼
              GPT-4.1-mini (doc_generator)
                        │
                        ▼
              data/docs/[módulo]/[tópico].md
                        │
                        ▼
              Chunking + Embeddings
                        │
                        ▼
              vectorstore/ (FAISS)
                        │
                        ▼
              LangGraph Agent ←→ Streamlit UI
```

## Variáveis de Ambiente

**Obrigatória para rodar o agente:**

| Variável | Descrição |
|---|---|
| `AI_PROVIDER` | Provedor de IA: `openai` (padrão) ou `gemini` (gratuito) |
| `OPENAI_API_KEY` | Chave da API OpenAI — necessária se `AI_PROVIDER=openai` |
| `GEMINI_API_KEY` | Chave da API Gemini — necessária se `AI_PROVIDER=gemini`. Obtenha gratuitamente em [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

**Opcionais (possuem valor padrão):**

| Variável | Padrão | Descrição |
|---|---|---|
| `VECTORSTORE_PATH` | `vectorstore` | Onde está o índice FAISS |
| `AGENT_MODEL` | `gpt-4.1-mini` | Modelo para o agente RAG |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings |

**Necessárias apenas para reprocessar o conteúdo do zero:**

| Variável | Descrição |
|---|---|
| `COURSE_PATH` | Pasta raiz com os arquivos de conteúdo (`.vtt`/`.pdf`) |
| `DOCS_OUTPUT_PATH` | Onde salvar os Markdowns gerados (padrão: `data/docs`) |
| `DOC_GEN_MODEL` | Modelo para geração de documentação (padrão: `gpt-4.1-mini`) |
