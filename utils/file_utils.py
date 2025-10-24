import requests
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


def download_pdf(url: str, output_dir: str, filename: str, skip_existing: bool = True) -> bool:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename

    # Vérifier si le fichier existe déjà
    if skip_existing and file_path.exists():
        file_size = file_path.stat().st_size
        if file_size > 1000:  # Au moins 1KB pour être valide
            logger.debug(f"Fichier déjà présent ({file_size / 1024:.1f} KB), téléchargement ignoré: {filename}")
            return True
        else:
            logger.warning(f"Fichier existant trop petit ({file_size} bytes), re-téléchargement: {filename}")
            file_path.unlink()  # Supprimer le fichier corrompu

    # Télécharger le fichier
    try:
        logger.debug(f"Début du téléchargement depuis: {url}")
        response = requests.get(url, timeout=15, stream=True)
        response.raise_for_status()

        # Vérifier le Content-Type
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
            logger.warning(f"Le fichier ne semble pas être un PDF (Content-Type: {content_type})")

        # Écrire le fichier
        with open(file_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_file.write(chunk)

        # Vérifier la taille du fichier téléchargé
        file_size = file_path.stat().st_size

        if file_size < 1000:
            logger.error(f"Fichier téléchargé trop petit ({file_size} bytes): {filename}")
            file_path.unlink()  # Supprimer le fichier invalide
            return False

        logger.info(f"PDF téléchargé avec succès ({file_size / 1024:.1f} KB): {filename}")
        return True

    except requests.exceptions.Timeout:
        logger.error(f"Timeout lors du téléchargement: {url}")
        return False

    except requests.exceptions.HTTPError as e:
        logger.error(f"Erreur HTTP {e.response.status_code} lors du téléchargement: {url}")
        return False

    except requests.exceptions.ConnectionError:
        logger.error(f"Erreur de connexion lors du téléchargement: {url}")
        return False

    except Exception as e:
        logger.error(f"Erreur inattendue lors du téléchargement de {url}")
        logger.exception(e)
        return False


def get_file_size(file_path: str) -> int:
    path = Path(file_path)
    if path.exists():
        size = path.stat().st_size
        logger.debug(f"Taille du fichier {file_path}: {size} bytes")
        return size
    logger.warning(f"Fichier non trouvé: {file_path}")
    return 0


def list_pdfs(directory: str) -> list:
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Dossier inexistant: {directory}")
        return []

    pdf_files = list(dir_path.glob("**/*.pdf"))
    logger.info(f"Trouvé {len(pdf_files)} fichier(s) PDF dans {directory}")

    return [str(f) for f in pdf_files]


def create_directory(directory: str) -> bool:
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Dossier créé/vérifié: {directory}")
        return True
    except Exception as e:
        logger.error(f"Impossible de créer le dossier {directory}")
        logger.exception(e)
        return False