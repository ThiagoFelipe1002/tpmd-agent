"""
Anonimiza os documentos Markdown em data/docs/:
- Substitui nomes de pessoas por termos genéricos
- Substitui nomes de empresas por termos genéricos
- Remove rodapés com referências ao curso/instrutor
"""

import re
from pathlib import Path

docs_dir = Path("data/docs")
md_files = list(docs_dir.rglob("*.md"))

# (padrão regex, substituição, flags)
REPLACEMENTS = [
    # Nome do instrutor (como sujeito: "Rui destaca", "Rui explica", "Rui apresenta")
    (r'\bRui\s+(destaca|explica|apresenta|ressalta|reforça|menciona|afirma|diz|comenta|pontua|enfatiza|compartilha|mostra|indica|sugere|recomenda|traz|aborda|esclarece|orienta|alerta|define|descreve|lembra|acrescenta|conclui|finaliza|demonstra|exemplifica|cita)\b',
     r'O especialista \1', re.IGNORECASE),
    # "Segundo Rui", "Para Rui", "De acordo com Rui"
    (r'\b(Segundo|Para|De acordo com|Na visão de|Na opinião de)\s+Rui\b',
     r'\1 o especialista', re.IGNORECASE),
    # Rui sozinho ou com pontuação
    (r'\bRui\b', 'o especialista', 0),
    # Parceiros/sócios
    (r'\bLeandro Ladeira\b', 'um especialista de referência', re.IGNORECASE),
    (r'\bLadeirinha\b', 'um especialista de referência', re.IGNORECASE),
    (r'\bLeandro\b', 'um especialista de referência', re.IGNORECASE),
    (r'\bVictor\b', 'um profissional da área', re.IGNORECASE),
    # Empresas
    (r'\bRed2Go\b', 'uma agência de tráfego pago', re.IGNORECASE),
    (r'\bCáteda Nascen\b', 'uma empresa do setor', re.IGNORECASE),
    (r'\bCateda Nascen\b', 'uma empresa do setor', re.IGNORECASE),
    (r'\bCátedra Nascen\b', 'uma empresa do setor', re.IGNORECASE),
    (r'\bNascen\b', 'uma empresa do setor', re.IGNORECASE),
    (r'\bCáteda\b', 'uma empresa do setor', re.IGNORECASE),
    # Rodapés automáticos — várias variações
    (r'\n?# Fim do Documento\s*\n---\s*\n.*?$', '', re.DOTALL),
    (r'\n?---\s*\n\s*# Fim do Documento.*?$', '', re.DOTALL),
    (r'\n?# Fim do Documento.*?$', '', re.DOTALL),
    (r'\n?---\n\n\*?Este documento foi elaborado.*?$', '', re.DOTALL),
    (r'\n?\*?Este documento foi elaborado.*?$', '', re.DOTALL),
    (r'\n?\*?Documento elaborado com base.*?$', '', re.DOTALL),
    # Linhas de referência ao curso (dentro de seções ## Referências)
    (r'^\s*[-*]\s*Transcrição da aula.*?\n', '', re.MULTILINE),
    (r'^\s*[-*]\s*Slides do (curso|módulo|material).*?\n', '', re.MULTILINE | re.IGNORECASE),
    (r'^\s*[-*]\s*Material base:.*?transcrição.*?\n', '', re.MULTILINE | re.IGNORECASE),
    # Nome real do instrutor
    (r'\bRuy Guimarães\b', 'o especialista', re.IGNORECASE),
    (r'\bRuy\b', 'o especialista', re.IGNORECASE),
    # Super ADS referência ao curso
    (r'"Super ADS"', 'o curso', 0),
    (r"'Super ADS'", 'o curso', 0),
    (r'\bSuper ADS\b', 'o curso', re.IGNORECASE),
]

modified = 0
for f in md_files:
    text = f.read_text(encoding="utf-8")
    new_text = text
    for pattern, repl, *flags in REPLACEMENTS:
        flag = flags[0] if flags else 0
        new_text = re.sub(pattern, repl, new_text, flags=flag)
    if new_text != text:
        f.write_text(new_text.rstrip() + "\n", encoding="utf-8")
        modified += 1

print(f"Modificados: {modified}/{len(md_files)}")

# Verificação final (case-sensitive para evitar falsos positivos como construir/ruim)
remaining = {}
checks = {
    r"\bRuy\b": "Ruy (instrutor)",
    r"\bRui\b": "Rui (nome próprio)",
    r"\bLeandro\b": "Leandro",
    r"\bRed2Go\b": "Red2Go",
    r"\bNascen\b": "Nascen",
    r"elaborado com base": "elaborado com base",
    r"transcrição da aula": "transcrição da aula",
    r"Ruy Guimarães": "Ruy Guimarães",
    r"Super ADS": "Super ADS",
}
for pattern, label in checks.items():
    count = sum(len(re.findall(pattern, f.read_text(encoding="utf-8"), re.IGNORECASE))
                for f in md_files)
    if count:
        remaining[label] = count

if remaining:
    print("Ainda restam:")
    for k, v in remaining.items():
        print(f"  {k}: {v}")
else:
    print("Todos os termos removidos com sucesso.")
