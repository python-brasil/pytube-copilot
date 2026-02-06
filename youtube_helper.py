#!/usr/bin/env python3
"""
YouTube Helper CLI - Analisa videos e gera conteudo otimizado para YouTube
Extrai audio, transcreve, cria roteiro, titulos, descricao, tags e prompt de thumbnail
"""

import os
import sys
import time
import argparse
import tempfile
from pathlib import Path
from datetime import datetime

# Configurar encoding UTF-8 para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text
from rich import box
from rich.align import Align
from openai import OpenAI
from pydub import AudioSegment

# Carregar variaveis de ambiente
load_dotenv()

# Configuracoes
MAX_FILE_SIZE_MB = 25
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.webm', '.mp4', '.mpeg', '.mpga', '.oga', '.ogg']
PROMPTS_DIR = Path(__file__).parent / "prompts"
OUTPUT_DIR = Path("youtube_analysis")

# Console
console = Console(force_terminal=True)


def get_openai_client():
    """Inicializa o cliente OpenAI"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Erro: OPENAI_API_KEY nao encontrada no arquivo .env[/red]")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def show_banner():
    """Exibe o banner"""
    banner = r"""
 __   __         _____      _           _    _      _
 \ \ / /__  _   |_   _|   _| |__   ___ | |  | | ___| |_ __   ___ _ __
  \ V / _ \| | | || || | | | '_ \ / _ \| |__| |/ _ \ | '_ \ / _ \ '__|
   | | (_) | |_| || || |_| | |_) |  __/|  __  |  __/ | |_) |  __/ |
   |_|\___/ \__,_||_| \__,_|_.__/ \___||_|  |_|\___|_| .__/ \___|_|
                                                     |_|
    """
    console.print()
    console.print(Panel(
        Align.center(Text(banner, style="bold cyan")),
        title="[bold white]Analise de Videos para YouTube[/bold white]",
        subtitle="[dim]Powered by GPT-4o-mini & Whisper[/dim]",
        border_style="cyan",
        box=box.DOUBLE,
        padding=(0, 2)
    ))
    console.print()


def load_prompt(prompt_name: str) -> str:
    """Carrega um prompt do diretorio de prompts"""
    prompt_path = PROMPTS_DIR / f"{prompt_name}.txt"
    if not prompt_path.exists():
        console.print(f"[red]Erro: Prompt '{prompt_name}' nao encontrado em {PROMPTS_DIR}[/red]")
        sys.exit(1)
    return prompt_path.read_text(encoding="utf-8")


def get_file_size_mb(file_path: Path) -> float:
    """Retorna o tamanho do arquivo em MB"""
    return file_path.stat().st_size / (1024 * 1024)


def extract_audio(video_path: Path) -> Path:
    """Extrai audio de um arquivo de video usando pydub"""
    console.print("[cyan]Extraindo audio do video...[/cyan]")

    temp_dir = Path(tempfile.mkdtemp(prefix="yt_helper_"))
    audio_path = temp_dir / "audio.mp3"

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("[cyan]Processando video...", total=None)

            # Carregar video e extrair audio
            audio = AudioSegment.from_file(str(video_path))
            audio.export(str(audio_path), format="mp3", bitrate="128k")

        console.print(f"[green][OK] Audio extraido: {get_file_size_mb(audio_path):.2f} MB[/green]")
        return audio_path

    except Exception as e:
        console.print(f"[red]Erro ao extrair audio: {e}[/red]")
        console.print("[yellow]Dica: Certifique-se de ter o FFmpeg instalado no sistema[/yellow]")
        sys.exit(1)


def split_audio_for_transcription(file_path: Path, chunk_duration_ms: int = 600000) -> list:
    """Divide o audio em chunks menores para transcricao

    Retorna uma lista de tuplas (chunk_path, offset_seconds) onde offset_seconds
    e o tempo de inicio do chunk no audio original
    """
    console.print("[yellow]Arquivo grande. Dividindo em partes...[/yellow]")

    audio = AudioSegment.from_file(str(file_path))
    total_duration = len(audio)
    chunks = []
    temp_dir = file_path.parent

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Dividindo audio...", total=total_duration)

        start = 0
        part_num = 1

        while start < total_duration:
            end = min(start + chunk_duration_ms, total_duration)
            chunk = audio[start:end]

            chunk_path = temp_dir / f"chunk_{part_num:03d}.mp3"
            chunk.export(str(chunk_path), format="mp3")
            # Armazena o caminho e o offset em segundos
            chunks.append((chunk_path, start / 1000.0))

            progress.update(task, completed=end)
            start = end
            part_num += 1

    console.print(f"[green][OK] Audio dividido em {len(chunks)} partes[/green]")
    return chunks


def transcribe_audio(client: OpenAI, file_path: Path, language: str = "pt") -> dict:
    """Transcreve um arquivo de audio usando Whisper com timestamps

    Retorna um dicionario com:
    - 'text': texto completo da transcricao
    - 'segments': lista de segmentos com timestamps (start, end, text)
    """
    file_size_mb = get_file_size_mb(file_path)
    all_segments = []
    transcription_parts = []

    if file_size_mb > MAX_FILE_SIZE_MB:
        chunks = split_audio_for_transcription(file_path)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Transcrevendo...", total=len(chunks))

            for i, (chunk_path, offset_seconds) in enumerate(chunks, 1):
                progress.update(task, description=f"[cyan]Transcrevendo parte {i}/{len(chunks)}...")

                try:
                    with open(chunk_path, "rb") as audio_file:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language=language,
                            response_format="verbose_json",
                            timestamp_granularities=["segment"]
                        )

                    transcription_parts.append(response.text)

                    # Ajustar timestamps com o offset do chunk
                    if hasattr(response, 'segments') and response.segments:
                        for seg in response.segments:
                            all_segments.append({
                                'start': seg.start + offset_seconds,
                                'end': seg.end + offset_seconds,
                                'text': seg.text.strip()
                            })

                except Exception as e:
                    console.print(f"[red]Erro na parte {i}: {e}[/red]")
                    transcription_parts.append(f"[ERRO NA PARTE {i}]")

                progress.update(task, advance=1)
                chunk_path.unlink()  # Limpar chunk
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("[cyan]Transcrevendo audio...", total=None)

            with open(file_path, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )

            transcription_parts.append(response.text)

            # Processar segmentos
            if hasattr(response, 'segments') and response.segments:
                for seg in response.segments:
                    all_segments.append({
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text.strip()
                    })

    return {
        'text': "\n\n".join(transcription_parts),
        'segments': all_segments
    }


def call_agent(client: OpenAI, system_prompt: str, user_content: str, agent_name: str) -> str:
    """Chama um agente GPT-4o-mini com o prompt especificado"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task(f"[cyan]{agent_name} trabalhando...", total=None)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=4000
        )

    return response.choices[0].message.content


def format_duration(seconds: float) -> str:
    """Formata duracao em HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float) -> str:
    """Formata timestamp em MM:SS ou HH:MM:SS para YouTube"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_transcription_with_timestamps(segments: list) -> str:
    """Formata a transcricao com timestamps por segmento

    Formato:
    [00:00] Texto do segmento...
    [00:15] Proximo segmento...
    """
    if not segments:
        return ""

    lines = []
    for seg in segments:
        timestamp = format_timestamp(seg['start'])
        text = seg['text']
        lines.append(f"[{timestamp}] {text}")

    return "\n".join(lines)


def generate_timestamped_sections(segments: list, total_duration: float) -> str:
    """Gera uma versao agrupada da transcricao com timestamps a cada ~30 segundos

    Util para ter uma visao geral do conteudo por momento do video
    """
    if not segments:
        return ""

    # Agrupar segmentos em intervalos de 30 segundos
    interval = 30.0
    sections = []
    current_section_start = 0.0
    current_section_text = []

    for seg in segments:
        # Se o segmento pertence a um novo intervalo, salva o anterior
        while seg['start'] >= current_section_start + interval:
            if current_section_text:
                timestamp = format_timestamp(current_section_start)
                text = " ".join(current_section_text)
                sections.append(f"[{timestamp}] {text}")
            current_section_start += interval
            current_section_text = []

        current_section_text.append(seg['text'])

    # Adicionar ultimo segmento
    if current_section_text:
        timestamp = format_timestamp(current_section_start)
        text = " ".join(current_section_text)
        sections.append(f"[{timestamp}] {text}")

    return "\n\n".join(sections)


def get_audio_duration(file_path: Path) -> float:
    """Retorna a duracao do audio em segundos"""
    try:
        audio = AudioSegment.from_file(str(file_path))
        return len(audio) / 1000
    except Exception:
        return 0


def save_analysis(video_path: Path, transcription_data: dict, roteiro: str,
                  titulos_desc_tags: str, thumbnail: str, total_duration: float) -> Path:
    """Salva toda a analise em um unico arquivo"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{video_path.stem}_analise_{timestamp}.txt"
    output_path = OUTPUT_DIR / output_name

    # Formatar transcricao com timestamps
    transcription_text = transcription_data['text']
    segments = transcription_data['segments']

    # Transcricao detalhada com timestamp por segmento
    timestamped_transcription = format_transcription_with_timestamps(segments)

    # Versao agrupada por intervalos de 30s
    sectioned_transcription = generate_timestamped_sections(segments, total_duration)

    content = f"""{'='*80}
                    YOUTUBE HELPER - ANALISE COMPLETA
{'='*80}

Arquivo: {video_path.name}
Data: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
Duracao Total: {format_duration(total_duration)}

{'='*80}
                    TRANSCRICAO COM TIMESTAMPS (DETALHADA)
{'='*80}

{timestamped_transcription}

{'='*80}
                    TRANSCRICAO AGRUPADA (A CADA 30s)
{'='*80}

{sectioned_transcription}

{'='*80}
                         TRANSCRICAO ORIGINAL (TEXTO)
{'='*80}

{transcription_text}

{'='*80}
                              ROTEIRO
{'='*80}

{roteiro}

{'='*80}
                    TITULOS, DESCRICAO E TAGS
{'='*80}

{titulos_desc_tags}

{'='*80}
                       PROMPT PARA THUMBNAIL
{'='*80}

{thumbnail}

{'='*80}
              Gerado por YouTube Helper - VelociScribe
{'='*80}
"""

    output_path.write_text(content, encoding="utf-8")
    return output_path


def analyze_video(video_path: Path, language: str = "pt"):
    """Fluxo principal de analise do video"""
    client = get_openai_client()
    start_time = time.time()

    # Verificar se e video ou audio
    is_video = video_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS

    if is_video:
        # Extrair audio do video
        audio_path = extract_audio(video_path)
    else:
        audio_path = video_path

    # Mostrar info do arquivo
    duration = get_audio_duration(audio_path)
    console.print(Panel(
        f"[white]Arquivo:[/white] [cyan]{video_path.name}[/cyan]\n"
        f"[white]Duracao:[/white] [cyan]{format_duration(duration)}[/cyan]\n"
        f"[white]Tamanho:[/white] [cyan]{get_file_size_mb(audio_path):.2f} MB[/cyan]\n"
        f"[white]Idioma:[/white] [cyan]{language.upper()}[/cyan]",
        title="[bold]Informacoes do Arquivo[/bold]",
        border_style="blue"
    ))
    console.print()

    # ETAPA 1: Transcricao com Timestamps
    console.print(Panel("[bold]ETAPA 1/4: Transcricao com Timestamps[/bold]", border_style="yellow"))
    transcription_data = transcribe_audio(client, audio_path, language)
    transcription_text = transcription_data['text']
    num_segments = len(transcription_data['segments'])
    console.print(f"[green][OK] Transcricao concluida ({len(transcription_text.split())} palavras, {num_segments} segmentos com timestamps)[/green]")
    console.print()

    # ETAPA 2: Roteiro (inclui transcricao com timestamps para contexto)
    console.print(Panel("[bold]ETAPA 2/4: Gerando Roteiro[/bold]", border_style="yellow"))
    roteiro_prompt = load_prompt("roteirista")

    # Preparar conteudo com timestamps para o roteirista gerar capitulos precisos
    timestamped_content = format_transcription_with_timestamps(transcription_data['segments'])
    roteiro_input = f"""DURACAO TOTAL DO VIDEO: {format_duration(duration)}

TRANSCRICAO COM TIMESTAMPS:
{timestamped_content}

TEXTO COMPLETO:
{transcription_text}"""

    roteiro = call_agent(client, roteiro_prompt, roteiro_input, "Agente Roteirista")
    console.print("[green][OK] Roteiro gerado[/green]")
    console.print()

    # ETAPA 3: Titulos, Descricao e Tags
    console.print(Panel("[bold]ETAPA 3/4: Gerando Titulos, Descricao e Tags[/bold]", border_style="yellow"))
    seo_prompt = load_prompt("titulo_descricao_tags")
    seo_content = f"DURACAO DO VIDEO: {format_duration(duration)}\n\nTRANSCRICAO:\n{transcription_text}\n\nROTEIRO:\n{roteiro}"
    titulos_desc_tags = call_agent(client, seo_prompt, seo_content, "Agente SEO")
    console.print("[green][OK] Titulos, descricao e tags gerados[/green]")
    console.print()

    # ETAPA 4: Thumbnail
    console.print(Panel("[bold]ETAPA 4/4: Gerando Prompt de Thumbnail[/bold]", border_style="yellow"))
    thumb_prompt = load_prompt("thumbnail")
    thumb_content = f"ROTEIRO:\n{roteiro}\n\nTITULOS E DESCRICAO:\n{titulos_desc_tags}"
    thumbnail = call_agent(client, thumb_prompt, thumb_content, "Agente Thumbnail")
    console.print("[green][OK] Prompt de thumbnail gerado[/green]")
    console.print()

    # Limpar arquivo de audio temporario se foi extraido de video
    if is_video and audio_path.exists():
        audio_path.unlink()
        audio_path.parent.rmdir()

    # Salvar resultado
    output_path = save_analysis(video_path, transcription_data, roteiro, titulos_desc_tags, thumbnail, duration)

    elapsed_time = time.time() - start_time

    console.print(Panel(
        f"[green][OK] Analise concluida com sucesso![/green]\n\n"
        f"[white]Tempo total:[/white] [cyan]{elapsed_time:.1f} segundos[/cyan]\n"
        f"[white]Segmentos com timestamp:[/white] [cyan]{num_segments}[/cyan]\n"
        f"[white]Arquivo salvo em:[/white]\n[cyan]{output_path}[/cyan]",
        title="[bold green]Resultado[/bold green]",
        border_style="green"
    ))


def main():
    """Ponto de entrada do CLI"""
    parser = argparse.ArgumentParser(
        description="YouTube Helper - Analisa videos e gera conteudo otimizado para YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python youtube_helper.py video.mp4
  python youtube_helper.py audio.mp3 --language en
  python youtube_helper.py "C:\\Videos\\meu video.mp4" -l pt

Formatos suportados:
  Video: .mp4, .mkv, .avi, .mov, .webm, .flv, .wmv
  Audio: .mp3, .wav, .m4a, .webm, .ogg
        """
    )

    parser.add_argument(
        "file",
        type=str,
        help="Caminho do arquivo de video ou audio"
    )

    parser.add_argument(
        "-l", "--language",
        type=str,
        default="pt",
        choices=["pt", "en", "es", "fr", "de", "it", "ja", "ko", "zh"],
        help="Idioma do audio (padrao: pt)"
    )

    args = parser.parse_args()

    # Processar caminho do arquivo
    file_path = Path(args.file.strip('"').strip("'"))

    if not file_path.exists():
        console.print(f"[red]Erro: Arquivo nao encontrado: {file_path}[/red]")
        sys.exit(1)

    # Verificar formato
    all_formats = SUPPORTED_VIDEO_FORMATS + SUPPORTED_AUDIO_FORMATS
    if file_path.suffix.lower() not in all_formats:
        console.print(f"[red]Erro: Formato nao suportado: {file_path.suffix}[/red]")
        console.print(f"[yellow]Formatos suportados: {', '.join(all_formats)}[/yellow]")
        sys.exit(1)

    try:
        show_banner()
        analyze_video(file_path, args.language)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operacao cancelada pelo usuario.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Erro: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
