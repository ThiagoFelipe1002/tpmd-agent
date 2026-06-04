"""
Gerador de documentação Markdown rica por tópico.
Combina transcrição (VTT) + slides (PDFs) e usa GPT para gerar
um documento estruturado e rico em informações.
"""

from pathlib import Path

from openai import OpenAI

from src.ingest.vtt_parser import parse_vtt
from src.ingest.pdf_extractor import extract_all_pdfs_in_folder


SYSTEM_PROMPT = """Você é um especialista em criar documentação técnica rica e precisa sobre tráfego pago e marketing digital.
Sua tarefa é transformar uma transcrição de vídeo e conteúdo de slides em um documento Markdown extremamente detalhado, organizado e útil.

O documento deve:
- Capturar TODOS os conceitos, frameworks, estratégias e metodologias mencionados
- Organizar o conteúdo em seções claras com hierarquia de títulos
- Preservar exemplos práticos, casos de uso e analogias usadas
- Destacar termos técnicos, métricas e nomenclaturas específicas da área
- Incluir listas de pontos-chave onde apropriado
- Ser fiel ao conteúdo original sem omitir informações relevantes
- Usar linguagem clara e direta
- Incluir uma seção de "Resumo" no início e "Pontos-Chave" ao final

NÃO invente informações. Use APENAS o conteúdo fornecido.
NÃO mencione nomes de criadores, autores, instrutores ou a origem do material."""


USER_PROMPT_TEMPLATE = """# Tópico: {module_name} / {lesson_name}

## Transcrição
{transcript}

## Conteúdo dos Slides
{slides_content}

---
Gere um documento Markdown rico, estruturado e completo sobre este conteúdo, capturando toda a informação relevante."""


def generate_lesson_doc(
    lesson_folder: Path,
    client: OpenAI,
    model: str = "gpt-4.1-mini",
) -> str:
    """
    Gera documentação Markdown para uma pasta de lição.

    Args:
        lesson_folder: Caminho da pasta da lição
        client: Cliente OpenAI inicializado
        model: Modelo a ser usado

    Returns:
        Documento Markdown gerado.
    """
    module_name = lesson_folder.parent.name
    lesson_name = lesson_folder.name

    # Extrai transcrição VTT
    vtt_files = list(lesson_folder.glob("*.vtt"))
    transcript = ""
    if vtt_files:
        transcript = parse_vtt(vtt_files[0])

    # Extrai conteúdo dos PDFs
    pdf_texts = extract_all_pdfs_in_folder(lesson_folder)
    slides_parts: list[str] = []
    for pdf_name, pdf_text in pdf_texts.items():
        slides_parts.append(f"### Arquivo: {pdf_name}\n{pdf_text}")
    slides_content = "\n\n".join(slides_parts) if slides_parts else "(Sem slides para esta aula)"

    # Se não há conteúdo algum, retorna vazio
    if not transcript and not slides_parts:
        return ""

    prompt = USER_PROMPT_TEMPLATE.format(
        module_name=module_name,
        lesson_name=lesson_name,
        transcript=transcript or "(Sem transcrição disponível)",
        slides_content=slides_content,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def collect_lesson_folders(course_root: str | Path) -> list[Path]:
    """
    Coleta todas as pastas de lições que contêm pelo menos um .vtt ou .pdf.

    Args:
        course_root: Pasta raiz do curso

    Returns:
        Lista ordenada de pastas de lições.
    """
    course_root = Path(course_root)
    lesson_folders: list[Path] = []

    for item in sorted(course_root.rglob("*")):
        if item.is_dir():
            has_vtt = any(item.glob("*.vtt"))
            has_pdf = any(item.glob("*.pdf"))
            if has_vtt or has_pdf:
                lesson_folders.append(item)

    return lesson_folders
